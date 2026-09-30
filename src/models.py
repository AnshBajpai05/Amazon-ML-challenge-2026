"""GBDT training on shared S1-grouped folds with fold-averaged test inference (plan section 7).

Backends: LightGBM (CPU, deterministic) or XGBoost (CUDA when a working GPU is found).
Every learned stage (meta-blocker, pass-1, pass-2, entity model) reuses the same folds,
so no stage sees labels of the rows it is later evaluated on.
"""
from __future__ import annotations

import json
import logging
import shutil
import time

import numpy as np

from .compat import effective_cpus

log = logging.getLogger("ber")


def _threads():
    return effective_cpus()


def probe_xgb_cuda():
    if shutil.which("nvidia-smi") is None:
        return False, "nvidia-smi not found"
    try:
        import xgboost as xgb
        rng = np.random.RandomState(0)
        X = rng.rand(512, 5).astype(np.float32)
        y = (X[:, 0] > 0.5).astype(np.float32)
        bst = xgb.train({"device": "cuda", "tree_method": "hist", "objective": "binary:logistic",
                         "verbosity": 0}, xgb.DMatrix(X, label=y), 3)
        dev = json.loads(bst.save_config())["learner"]["generic_param"].get("device", "")
        if "cuda" not in dev:
            return False, f"xgboost fell back to {dev!r}"
        p = bst.inplace_predict(X)
        if not np.isfinite(p).all():
            return False, "non-finite predictions"
        return True, "ok"
    except Exception as e:  # no CUDA build, unsupported GPU arch (e.g. sm_60), driver issues
        return False, repr(e)[:300]


def detect_backend(pref: str) -> str:
    if pref in ("lightgbm", "xgboost_cpu"):
        return pref
    ok, why = probe_xgb_cuda()
    if ok:
        return "xgboost_cuda"
    if pref == "xgboost_cuda":
        log.warning("xgboost CUDA unavailable (%s); falling back to LightGBM", why)
    else:
        log.info("GPU GBDT unavailable (%s); using LightGBM on CPU", why)
    return "lightgbm"


def _train_one(backend, Xtr, ytr, Xva, yva, feat_names, params, seed, monotone, rounds, es):
    if backend == "lightgbm":
        import lightgbm as lgb
        p = dict(objective="binary", deterministic=True, force_row_wise=True, seed=seed,
                 num_threads=_threads(), verbose=-1, **params)
        if monotone and any(monotone):
            p["monotone_constraints"] = list(monotone)
        dtr = lgb.Dataset(Xtr, ytr, feature_name=feat_names, free_raw_data=True)
        dva = lgb.Dataset(Xva, yva, reference=dtr)
        bst = lgb.train(p, dtr, num_boost_round=rounds, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(es, verbose=False)])
        it = bst.best_iteration or bst.current_iteration()
        pred = lambda X: bst.predict(X, num_iteration=it)
        imp = dict(zip(feat_names, bst.feature_importance("gain").astype(float)))
        return pred, imp, it, ("lightgbm", bst, it)
    import xgboost as xgb
    p = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", seed=seed,
             device="cuda" if backend == "xgboost_cuda" else "cpu", nthread=_threads(), verbosity=0, **params)
    if monotone and any(monotone):
        p["monotone_constraints"] = "(" + ",".join(str(int(m)) for m in monotone) + ")"
    dtr = xgb.QuantileDMatrix(Xtr, ytr, feature_names=feat_names, max_bin=p.get("max_bin", 256))
    dva = xgb.QuantileDMatrix(Xva, yva, feature_names=feat_names, ref=dtr)
    bst = xgb.train(p, dtr, num_boost_round=rounds, evals=[(dva, "va")], early_stopping_rounds=es,
                    verbose_eval=False)
    it = int(getattr(bst, "best_iteration", rounds - 1)) + 1
    pred = lambda X: bst.inplace_predict(X, iteration_range=(0, it))
    imp = {k: float(v) for k, v in bst.get_score(importance_type="total_gain").items()}
    return pred, imp, it, ("xgboost", bst, it)


def fit_oof(X, y, folds, X_test, feat_names, cfg, backend, seeds, monotone=None, name="model",
            params=None, rounds=None, es=None, ckpt=None, ckpt_key=None, train_mask=None, keep_models=False):
    """Out-of-fold predictions on (X, y) with the shared folds, plus the mean of all fold models on X_test.
    X, X_test: float32 arrays. folds: fold id per row. Returns (oof, test_pred, info).
    ckpt/ckpt_key: optional cache.Checkpointer; each finished fold is checkpointed, so an interrupted
    run resumes at the next fold. keep_models: return the fitted fold models in info["_models"] so that
    large test sets can be predicted later, chunk by chunk (predict_models)."""
    t0 = time.time()
    mcfg = cfg["model"]
    if params is None:
        params = dict(mcfg["lgb"] if backend == "lightgbm" else mcfg["xgb"])
    rounds = rounds or mcfg["num_boost_round"]
    es = es or mcfg["early_stopping"]
    y = np.asarray(y, np.float32)
    oof = np.zeros(len(X), np.float64)
    tp = np.zeros(len(X_test), np.float64) if X_test is not None else None
    imp = {f: 0.0 for f in feat_names}
    iters, n_models, fold_stats, models = [], 0, [], []
    ufolds = np.unique(folds)
    log.info("  %s [%s]: %d rows x %d feats, pos rate %.4f, %d folds x %d seeds", name, backend, len(X),
             X.shape[1], float(y.mean()) if len(y) else 0.0, len(ufolds), len(seeds))
    for fi, f in enumerate(ufolds):
        tf = time.time()
        tr, va = folds != f, folds == f
        if train_mask is not None:                   # e.g. negative subsampling: fit on a subset, predict all
            tr = tr & train_mask
        if y[tr].min() == y[tr].max():               # degenerate fold: constant prior
            oof[va] = y[tr].mean()
            if tp is not None:
                tp += y[tr].mean()
            models.append(("const", float(y[tr].mean()), None))
            n_models += 1
            continue
        fkey = f"{ckpt_key}-fold{int(f)}" if (ckpt is not None and ckpt_key) else None
        saved = ckpt.load(fkey) if fkey else None
        if saved is None:
            o_va = np.zeros(int(va.sum()))
            t_part = np.zeros(len(X_test)) if tp is not None else None
            f_imp, fit_iters, f_models = {}, [], []
            for sd in seeds:
                pred, im, it, mdl = _train_one(backend, X[tr], y[tr], X[va], y[va], feat_names, params, int(sd),
                                               monotone, rounds, es)
                if keep_models:
                    f_models.append(mdl)
                o_va += pred(X[va]) / len(seeds)
                if t_part is not None and len(X_test):
                    t_part += pred(X_test) / len(seeds)
                for k, v in im.items():
                    f_imp[k] = f_imp.get(k, 0.0) + v
                fit_iters.append(it)
            saved = {"o_va": o_va, "t_part": t_part, "imp": f_imp, "iters": fit_iters, "models": f_models}
            if fkey:
                ckpt.save(fkey, saved)
        oof[va] = saved["o_va"]
        if tp is not None and saved["t_part"] is not None:
            tp += saved["t_part"]
        for k, v in saved["imp"].items():
            imp[k] = imp.get(k, 0.0) + v
        fit_iters = saved["iters"]
        iters.extend(fit_iters)
        if keep_models:
            models.append(("avg", saved.get("models", []), None))
        n_models += 1
        st = {"fold": int(f), "iters": fit_iters, "seconds": round(time.time() - tf, 1), **binary_metrics(y[va], oof[va])}
        fold_stats.append(st)
        done = fi + 1
        eta = (time.time() - t0) / done * (len(ufolds) - done)
        log.info("    %s fold %d/%d: AUC %.5f  AP %.5f  logloss %.5f  iters %s  %.0fs  (ETA %.0fs)", name, done,
                 len(ufolds), st.get("auc", float("nan")), st.get("ap", float("nan")),
                 st.get("logloss", float("nan")), fit_iters, st["seconds"], eta)
    if tp is not None:
        tp /= max(n_models, 1)
    tot = sum(imp.values()) or 1.0
    imp = dict(sorted(((k, v / tot) for k, v in imp.items()), key=lambda kv: -kv[1]))
    info = {"name": name, "backend": backend, "seconds": round(time.time() - t0, 1), "iterations": iters,
            "n_rows": int(len(X)), "n_features": int(X.shape[1]), "pos_rate": float(y.mean()) if len(y) else 0.0,
            "oof_metrics": binary_metrics(y, oof), "folds": fold_stats, "_models": models,
            "importance_top30": {k: round(v, 5) for k, v in list(imp.items())[:30]},
            "_importance_all": imp}
    log.info("  %s done: OOF AUC %.5f  AP %.5f  logloss %.5f  in %.1fs", name,
             info["oof_metrics"].get("auc", float("nan")), info["oof_metrics"].get("ap", float("nan")),
             info["oof_metrics"].get("logloss", float("nan")), time.time() - t0)
    return oof, tp, info


def predict_models(models, X):
    """Mean prediction of the fold models kept by fit_oof(keep_models=True); each fold averages its seeds."""
    out = np.zeros(len(X), np.float64)
    for kind, obj, _ in models:
        if kind == "const":
            out += obj
            continue
        part = np.zeros(len(X), np.float64)
        for backend, bst, it in obj:
            if backend == "lightgbm":
                part += bst.predict(X, num_iteration=it)
            else:
                part += bst.inplace_predict(X, iteration_range=(0, it))
        out += part / max(len(obj), 1)
    return out / max(len(models), 1)


def binary_metrics(y, p):
    """AUC / average precision / logloss / Brier; NaN when a class is missing."""
    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
    y = np.asarray(y)
    p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7)
    out = {"n": int(len(y)), "pos": int(y.sum())}
    if len(y) == 0 or y.min() == y.max():
        return out
    out.update(auc=float(roc_auc_score(y, p)), ap=float(average_precision_score(y, p)),
               logloss=float(log_loss(y, p)), brier=float(np.mean((p - y) ** 2)))
    return out


def calibration_report(y, p, bins=10):
    """Expected calibration error + reliability table (equal-width bins)."""
    y = np.asarray(y, np.float64)
    p = np.asarray(p, np.float64)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        mp, fp = float(p[m].mean()), float(y[m].mean())
        ece += m.mean() * abs(mp - fp)
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()), "mean_pred": round(mp, 4),
                     "frac_pos": round(fp, 4)})
    return {"ece": float(ece), "brier": float(np.mean((p - y) ** 2)) if len(y) else float("nan"),
            "reliability": rows}


def fit_isotonic(p, y):
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
    ir.fit(np.asarray(p, np.float64), np.asarray(y, np.float64))
    return ir


def make_folds(strata, n_folds, seed):
    """Fold id per S1, stratified by country x match-count bucket (each S1 is its own group)."""
    from sklearn.model_selection import StratifiedKFold
    strata = np.asarray(strata)
    n = len(strata)
    folds = np.zeros(n, np.int64)
    if n < n_folds:
        return np.arange(n) % max(n_folds, 1)
    vals, counts = np.unique(strata, return_counts=True)
    rare = set(vals[counts < n_folds])
    s = np.array(["__rare__" if x in rare else x for x in strata], dtype=object)
    if (s == "__rare__").sum() and (s == "__rare__").sum() < n_folds:
        s[s == "__rare__"] = vals[np.argmax(counts)]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for f, (_, va) in enumerate(skf.split(np.zeros(n), s)):
        folds[va] = f
    return folds
