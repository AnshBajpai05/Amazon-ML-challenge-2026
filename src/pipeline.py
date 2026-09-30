"""End-to-end: fit on a labelled split, predict an unlabelled split (plan section 0).

normalize -> blocking (per partition) -> meta-blocker (OOF) -> pair features -> pass-1 GBDT (OOF)
-> collective features -> [cross-encoder] -> pass-2 GBDT (OOF) -> entity model (OOF)
-> nested-CV decoder tuning -> decode test.
All learned stages share one fold assignment per S1, so no stage leaks labels into another.
Every stage is checkpointed (cache.py): an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import blocking as B
from .cache import NULL, code_hash, split_fingerprint
from .collective import COLLECTIVE, ENTITY_FEATS, add_collective, entity_features
from .decode import decode, selections, tune_decoder
from .features import monotone_vector, pair_features
from .metrics import blocking_kpis, evaluate, f05_entity
from .models import binary_metrics, calibration_report, detect_backend, fit_oof, make_folds
from .normalize import normalize_frame
from .track import Tracker, fmt_dur, write_table

log = logging.getLogger("ber")

KEY_FEATS = ["n_sd", "n_mu_a", "n_mu_b", "a_sd", "n_cos_c", "a_cos_c", "j_cos_c", "n_tset", "a_tset",
             "post_eq2", "post_conf2", "num_conf", "legal_conf", "n_ch", "meta"]


def _gold_keys(gold, s1_ids, pool_ids):
    s1_pos = {s: i for i, s in enumerate(s1_ids)}
    pool_pos = {s: i for i, s in enumerate(pool_ids)}
    n2 = len(pool_ids)
    keys = [s1_pos[s] * n2 + pool_pos[c] for s, lst in gold.items() if s in s1_pos for c in lst if c in pool_pos]
    G = np.zeros(len(s1_ids), np.int64)
    for s, lst in gold.items():
        if s in s1_pos:
            G[s1_pos[s]] = len(lst)
    return np.unique(np.asarray(keys, np.int64)), G


def _labels(pairs, gkeys, n2):
    return np.isin(pairs["s1"].to_numpy(np.int64) * n2 + pairs["cand"].to_numpy(np.int64), gkeys)


def _cand_map(pairs, n1, npool, order_col="meta"):
    """s1 id -> candidate ids (sorted by order_col desc)."""
    s1_ids, pool_ids = n1["id"].to_numpy(object), npool["id"].to_numpy(object)
    if not len(pairs):
        return {}
    d = pairs[["s1", "cand", order_col]].sort_values(["s1", order_col, "cand"], ascending=[True, False, True],
                                                     kind="stable")
    out = {}
    for s, grp in d.groupby("s1", sort=False)["cand"]:
        out[s1_ids[s]] = pool_ids[grp.to_numpy()].tolist()
    return out


def _partition_mode(cfg, n1, npool, gold, report):
    mode = cfg.get("partition", "auto")
    c1 = dict(zip(n1["id"], n1["cty"]))
    cp = dict(zip(npool["id"], npool["cty"]))
    same = [c1[s] == cp[c] for s, lst in gold.items() if s in c1 for c in lst if c in cp]
    rate = float(np.mean(same)) if same else 1.0
    report["same_country_match_rate"] = rate
    if mode == "auto":
        mode = "country" if rate >= 0.999 else "none"
    log.info("partition mode: %s (same-country match rate %.4f)", mode, rate)
    return mode


def _neural_flags(cfg, report):
    nc = cfg["neural"]
    want_ce, want_dense = nc.get("cross_encoder", "off"), nc.get("dense", "off")
    status = {"ok": False, "why": "not requested"}
    if want_ce != "off" or want_dense != "off":
        from .neural import torch_cuda_status
        status = torch_cuda_status()
    report["gpu_torch"] = status
    log.info("torch GPU status: %s", status)
    ce = want_ce == "on" or (want_ce == "auto" and bool(status.get("ok")))
    dense = want_dense == "on" or (want_dense == "auto" and bool(status.get("ok")))
    if (ce or dense) and not status.get("ok"):
        log.warning("neural boosters forced 'on' without a usable GPU (%s): running on CPU, this is slow",
                    status.get("why"))
    elif want_ce == "auto" and not status.get("ok"):
        log.info("cross-encoder skipped: no usable CUDA device (%s)", status.get("why"))
    return ce, dense


def _error_samples(path, pred, gold, cand, n1, npool, per_bucket=50, seed=0):
    """Top losses per error-budget bucket with raw strings, for the error-analysis loop (plan block F)."""
    s1 = n1.set_index("id")[["raw_name", "raw_addr", "cty"]]
    pl = npool.set_index("id")[["raw_name", "raw_addr"]]
    show = lambda ids: " || ".join(f"{i}: {pl.at[i, 'raw_name']} @ {pl.at[i, 'raw_addr']}" for i in ids)
    rows = []
    for s, g in gold.items():
        p, c, g = set(pred.get(s, ())), set(cand.get(s, ())), set(g)
        loss = 1 - f05_entity(p, g)
        if loss <= 0:
            continue
        if not g:
            b = "singleton_false_merge"
        elif not p:
            b = "missed_entity_blocking" if not (g & c) else "missed_entity_model"
        elif not (p & g):
            b = "wrong_pick_only"
        elif p - g:
            b = "extra_false_positives"
        else:
            b = "missed_extra_matches"
        rows.append((b, round(loss, 4), s, s1.at[s, "cty"], s1.at[s, "raw_name"], s1.at[s, "raw_addr"],
                     show(sorted(p)), show(sorted(g)), show(sorted(g - c))))
    df = pd.DataFrame(rows, columns=["bucket", "loss", "s1_id", "country", "s1_name", "s1_address", "predicted",
                                     "gold", "gold_not_in_candidates"])
    if len(df):
        df = (df.sample(frac=1.0, random_state=seed).sort_values(["bucket", "loss"], ascending=[True, False],
                                                                  kind="stable")
              .groupby("bucket", sort=True).head(per_bucket))
    df.to_csv(path, sep="\t", index=False)


def _small_params(cfg, backend):
    return dict(cfg["model"]["lgb_small"] if backend == "lightgbm" else cfg["model"]["xgb_small"])


def _ce_estimate_sec(n_fit, n_score, nc, gpu):
    """Rough cross-encoder cost (base-size encoder): training steps + scoring throughput + model load."""
    per_model = min(n_fit / 2.0, nc["ce_max_train"])
    steps = 2 * nc["ce_epochs"] * np.ceil(per_model / nc["ce_batch"])
    if gpu:
        return steps / 7.0 + n_score / 1200.0 + 240.0
    return steps / 0.3 + n_score / 40.0 + 60.0


def fit_predict(train, test, cfg, report, test_gold=None, out_dir=None, tracker=None, ckpt=None):
    """train/test: dicts from io_utils.load_split (train with gold). Returns (pred, cand) id maps for test.
    out_dir: folder for analysis artifacts; tracker: track.Tracker; ckpt: cache.Checkpointer."""
    T = tracker or Tracker(report)
    ck = ckpt or NULL
    seed = cfg["seed"]
    gold = train["gold"]
    budget = float(cfg.get("time_budget_hours", 11.0)) * 3600
    fp = {"train": split_fingerprint(train), "test": split_fingerprint(test)}
    report["data_fingerprint"] = fp

    # ---------------- 0. normalize
    k_norm = ck.key("normalize", fp, code_hash("normalize"))
    with T.stage("normalize"):
        n1_tr, np_tr, n1_te, np_te = ck.cached(k_norm, lambda: (
            normalize_frame(train["s1"]), normalize_frame(train["pool"]),
            normalize_frame(test["s1"]), normalize_frame(test["pool"])))
    mode = _partition_mode(cfg, n1_tr, np_tr, gold, report)
    backend = detect_backend(cfg.get("backend", "auto"))
    seeds = cfg["seeds_gpu"] if backend == "xgboost_cuda" else cfg["seeds_cpu"]
    report["backend"], report["seeds"] = backend, list(seeds)
    log.info("GBDT backend: %s, seeds %s", backend, list(seeds))
    use_ce, use_dense = _neural_flags(cfg, report)
    report["neural_used"] = {"cross_encoder": bool(use_ce), "dense_c8": bool(use_dense)}
    gpu_ok = bool(report.get("gpu_torch", {}).get("ok"))

    n_s1_tr, n2_tr = len(n1_tr), len(np_tr)
    gkeys, G = _gold_keys(gold, n1_tr["id"].tolist(), np_tr["id"].tolist())
    strata = n1_tr["cty"].to_numpy(object) + "_" + np.minimum(G, 3).astype(str)
    fold_s1 = make_folds(strata, cfg["n_folds"], seed)
    country_tr = dict(zip(n1_tr["id"], n1_tr["cty"]))
    gold_id = {s: gold.get(s, []) for s in n1_tr["id"]}
    nc = cfg["neural"]

    # ---------------- 1. (dense) + blocking + meta-blocking; the post-meta bundle is one checkpoint
    k_dense = ck.key("dense", k_norm, nc.get("dense_model"), code_hash("neural")) if use_dense else None
    k_blk = ck.key("blocking", k_norm, cfg["blocking"], mode, k_dense, code_hash("blocking", "neural"))
    k_meta = ck.key("meta", k_blk, cfg["meta"], _small_params(cfg, backend), backend, seeds[:1], cfg["n_folds"],
                    seed, code_hash("models", "blocking", "metrics", "pipeline"))
    MB = ck.load(k_meta)
    if MB is None:
        dense_tr = dense_te = None
        dense_failed = False
        if use_dense:
            with T.stage("dense_embeddings"):
                try:
                    from .neural import dense_embed

                    def _emb():
                        return (dense_embed(n1_tr, nc), dense_embed(np_tr, nc)), (dense_embed(n1_te, nc),
                                                                                 dense_embed(np_te, nc))
                    dense_tr, dense_te = ck.cached(k_dense, _emb)
                except Exception as e:
                    log.warning("dense channel disabled: %r", e)
                    report["neural_used"]["dense_c8"] = False
                    report["neural_used"]["dense_error"] = repr(e)[:300]
                    dense_tr = dense_te = None
                    dense_failed = True
                    k_blk = ck.key("blocking", k_norm, cfg["blocking"], mode, None, code_hash("blocking", "neural"))
        with T.stage("blocking"):
            def _block():
                return (*B.run_blocking(n1_tr, np_tr, cfg, mode, dense_tr),
                        *B.run_blocking(n1_te, np_te, cfg, mode, dense_te))
            if cfg.get("cache", {}).get("save_union", False):
                blk = ck.cached(k_blk, _block)
            else:                                    # the union is the biggest object: recompute, don't store
                blk = ck.load(k_blk) or _block()
            U_tr, idf_tr, bst_tr, U_te, idf_te, bst_te = blk
        with T.stage("meta_blocking"):
            MB = _meta_stage(U_tr, U_te, n1_tr, np_tr, n1_te, np_te, gkeys, n2_tr, fold_s1, gold_id, country_tr,
                             cfg, backend, seeds, ck, k_meta)
            MB.update(idf_tr=idf_tr, idf_te=idf_te, partitions={"train": bst_tr, "test": bst_te})
            del U_tr, U_te
        if not dense_failed:                     # the key promises the dense channel; don't cache without it
            ck.save(k_meta, MB)
    else:
        log.info("blocking + meta-blocking restored from checkpoint")
    P_tr, P_te, y, cf = MB["P_tr"], MB["P_te"], MB["y"], MB["cf"]
    idf_tr, idf_te = MB["idf_tr"], MB["idf_te"]
    report["blocking"], report["blocking_partitions"] = MB["blocking"], MB["partitions"]
    if MB.get("meta_info") is not None:
        report["meta_blocker"] = MB["meta_info"]
    log.info("blocking: train PC union=%.4f final=%.4f | cand/S1 train=%.1f test=%.1f",
             report["blocking"]["train_union"]["PC"], report["blocking"]["train_final"]["PC"],
             len(P_tr) / max(n_s1_tr, 1), len(P_te) / max(len(n1_te), 1))

    # ---------------- 2. pair features
    k_feat = ck.key("features", k_meta, code_hash("features", "normalize", "blocking"))
    with T.stage("features"):
        X_tr, X_te = ck.cached(k_feat, lambda: (pair_features(P_tr, n1_tr, np_tr, idf_tr),
                                                pair_features(P_te, n1_te, np_te, idf_te)))
    feats = list(X_tr.columns)
    fold_pair = fold_s1[P_tr["s1"].to_numpy()]
    mono = monotone_vector(feats) if cfg["model"].get("monotone", True) else None

    # ---------------- 3. pass-1
    k_p1 = ck.key("pass1", k_feat, cfg["model"], backend, list(seeds), code_hash("models", "features"))
    with T.stage("pass1"):
        t_p1 = time.time()
        p1_oof, p1_te, info1 = ck.cached(k_p1, lambda: fit_oof(
            X_tr.to_numpy(np.float32), y, fold_pair, X_te.to_numpy(np.float32), feats, cfg, backend, seeds, mono,
            name="pass1", ckpt=ck, ckpt_key=k_p1))
        t_p1 = max(time.time() - t_p1, info1.get("seconds", 0.0))
    report.setdefault("models", {})["pass1"] = {k: v for k, v in info1.items() if not k.startswith("_")}

    k_col = ck.key("collective", k_p1, cfg["collective"], code_hash("collective"))
    with T.stage("collective"):
        C_tr, C_te = ck.cached(k_col, lambda: (add_collective(P_tr, p1_oof, np_tr, cfg["collective"]["top_m"]),
                                               add_collective(P_te, p1_te, np_te, cfg["collective"]["top_m"])))

    # ---------------- 4. optional cross-encoder feature (GPU), budget-aware
    ce_tr = ce_te = None
    k_ce = ck.key("ce", k_col, {k: v for k, v in nc.items() if k.startswith("ce_")}, code_hash("neural"))
    if use_ce and len(P_tr) and len(P_te):
        m_tr = (p1_oof >= nc["ce_min_p1"]) | (C_tr["rank_in_s1"].to_numpy() <= nc["ce_top_rank"])
        m_te = (p1_te >= nc["ce_min_p1"]) | (C_te["rank_in_s1"].to_numpy() <= nc["ce_top_rank"])
        est = _ce_estimate_sec(int(m_tr.sum()), int(m_tr.sum() + 2 * m_te.sum()), nc, gpu_ok)
        left = budget - (time.time() - T.t0)
        need = est + 3.0 * t_p1 + 600
        report["ce_budget"] = {"estimate_sec": round(est), "time_left_sec": round(left),
                               "needed_incl_rest_sec": round(need)}
        if need > left and ck.load(k_ce) is None:
            log.warning("cross-encoder skipped by the time budget: needs ~%s (+rest), %s left of %.1fh",
                        fmt_dur(est), fmt_dur(left), budget / 3600)
            report["neural_used"]["cross_encoder"] = False
            report["neural_used"]["ce_skipped"] = "time budget"
        else:
            with T.stage("cross_encoder"):
                log.info("cross-encoder: ~%s estimated (%d train / %d test pairs to score)", fmt_dur(est),
                         int(m_tr.sum()), int(m_te.sum()))
                try:
                    from .neural import cross_encoder_feature
                    ce_tr, ce_te = ck.cached(k_ce, lambda: cross_encoder_feature(
                        {"pairs": P_tr, "n1": n1_tr, "npool": np_tr, "p1": p1_oof,
                         "rank": C_tr["rank_in_s1"].to_numpy(), "y": y, "fold": fold_pair},
                        {"pairs": P_te, "n1": n1_te, "npool": np_te, "p1": p1_te,
                         "rank": C_te["rank_in_s1"].to_numpy()}, nc, seed, ckpt=ck, ckpt_key=k_ce))
                except Exception as e:
                    log.warning("cross-encoder failed, continuing without it: %r", e)
                    report["neural_used"]["cross_encoder"] = False
                    report["neural_used"]["ce_error"] = repr(e)[:500]

    # ---------------- 5. pass-2 + entity model + decoder (with / without the CE feature: kill rule)
    variants = [("base", None, None)]
    if ce_tr is not None:
        variants.append(("with_ce", ce_tr, ce_te))
    s1_tr, s1_te = P_tr["s1"].to_numpy(), P_te["s1"].to_numpy()
    h_label = np.zeros(n_s1_tr, np.float32)
    if len(s1_tr):
        np.maximum.at(h_label, s1_tr, y)
    has_tr = np.bincount(s1_tr, minlength=n_s1_tr) > 0
    has_te = np.bincount(s1_te, minlength=len(n1_te)) > 0
    results = {}
    for vname, ctr, cte in variants:
        k_p2 = ck.key("pass2", vname, k_col, k_ce if ctr is not None else None, cfg["model"], backend, list(seeds),
                      code_hash("models", "collective"))
        with T.stage(f"pass2_{vname}"):
            V = ck.load(k_p2)
            if V is None:
                X2_tr = pd.concat([X_tr, C_tr], axis=1)
                X2_te = pd.concat([X_te, C_te], axis=1)
                if ctr is not None:
                    X2_tr["ce"], X2_te["ce"] = ctr, cte
                f2 = list(X2_tr.columns)
                mono2 = (mono + [0] * (len(f2) - len(feats))) if mono else None
                p2_oof, p2_te, info2 = fit_oof(X2_tr.to_numpy(np.float32), y, fold_pair, X2_te.to_numpy(np.float32),
                                               f2, cfg, backend, seeds, mono2, name=f"pass2_{vname}", ckpt=ck,
                                               ckpt_key=k_p2)
                del X2_tr, X2_te
                E_tr = entity_features(P_tr, p2_oof, X_tr, C_tr, n1_tr, n_s1_tr, idf_tr)
                E_te = entity_features(P_te, p2_te, X_te, C_te, n1_te, len(n1_te), idf_te)
                h_oof, h_te = np.zeros(n_s1_tr), np.zeros(len(n1_te))
                if has_tr.any() and h_label[has_tr].min() != h_label[has_tr].max():
                    ho, ht, infoh = fit_oof(E_tr.to_numpy(np.float32)[has_tr], h_label[has_tr], fold_s1[has_tr],
                                            E_te.to_numpy(np.float32)[has_te], ENTITY_FEATS, cfg, backend, seeds,
                                            name=f"entity_{vname}", params=_small_params(cfg, backend),
                                            rounds=2000, es=100, ckpt=ck, ckpt_key=k_p2 + "-entity")
                    h_oof[has_tr], h_te[has_te] = ho, ht
                else:
                    infoh = {"note": "degenerate entity labels; h = 0"}
                V = dict(p2_oof=p2_oof, p2_te=p2_te, h_oof=h_oof, h_te=h_te, info2=info2, infoh=infoh)
                ck.save(k_p2, V)
            k_dec = ck.key("decoder", k_p2, cfg["decoder"], cfg["n_folds"], seed, code_hash("decode"))
            rep, fitted = ck.cached(k_dec, lambda: tune_decoder(s1_tr, P_tr["cand"].to_numpy(), V["p2_oof"], y,
                                                                V["h_oof"], h_label, fold_s1, G, n_s1_tr,
                                                                cfg["decoder"]))
        results[vname] = dict(V, rep=rep, fitted=fitted)
        log.info("variant %-8s nested OOF F0.5 = %.5f  (per fold: %s)", vname, rep["nested_f05"],
                 [round(x["f05"], 4) for x in rep["per_fold"]])
    chosen = "base"
    if "with_ce" in results:
        gain = results["with_ce"]["rep"]["nested_f05"] - results["base"]["rep"]["nested_f05"]
        report["ce_gain_nested_f05"] = gain
        if gain >= nc.get("ce_min_gain", 0.001):
            chosen = "with_ce"
        log.info("cross-encoder gain %.5f -> using %s", gain, chosen)
    R = results[chosen]
    imp_all = {"pass1": info1.get("_importance_all", {}), "pass2": R["info2"].get("_importance_all", {})}
    strip = lambda d: {k: v for k, v in d.items() if not str(k).startswith("_")}
    report["chosen_variant"] = chosen
    report["models"]["pass2"] = strip(R["info2"])
    report["models"]["entity"] = strip(R["infoh"])
    report["decoder"] = R["rep"]
    report["variants_nested_f05"] = {k: v["rep"]["nested_f05"] for k, v in results.items()}

    # ---------------- 6. OOF predictions with the final policy (breakdowns, error budget, calibration)
    with T.stage("oof_evaluation"):
        s1_ids_tr, pool_ids_tr = n1_tr["id"].to_numpy(object), np_tr["id"].to_numpy(object)
        k_o, IDX_o, P_o = decode(s1_tr, P_tr["cand"].to_numpy(), R["p2_oof"], R["h_oof"], n_s1_tr, cfg["decoder"],
                                 R["fitted"])
        cand_col_tr = P_tr["cand"].to_numpy()
        sel_o = selections(k_o, IDX_o)
        pred_oof = {s1_ids_tr[i]: pool_ids_tr[cand_col_tr[rows]].tolist() for i, rows in enumerate(sel_o)}
        report["oof"] = evaluate(pred_oof, gold_id, country_tr, cf)
        report["oof"]["nested_f05"] = R["rep"]["nested_f05"]
        p2_cal_oof = R["fitted"]["iso"].predict(R["p2_oof"]) if len(y) else np.zeros(0)
        report["calibration"] = {
            "pass1_oof": binary_metrics(y, p1_oof),
            "pass2_oof_raw": {**binary_metrics(y, R["p2_oof"]), **calibration_report(y, R["p2_oof"])},
            "pass2_oof_isotonic": calibration_report(y, p2_cal_oof),
            "entity_oof": binary_metrics(h_label[has_tr], R["h_oof"][has_tr]) if has_tr.any() else {},
        }
        log.info("OOF macro F0.5: nested %.5f | final policy (in-sample calibration) %.5f",
                 R["rep"]["nested_f05"], report["oof"]["macro_f05"])
        log.info("OOF by country: %s", {k: round(v["f05"], 5) for k, v in report["oof"].get("by_country", {}).items()})
        log.info("OOF by true-match count: %s",
                 {k: round(v["f05"], 5) for k, v in report["oof"].get("by_match_count", {}).items()})
        log.info("OOF error budget: %s", report["oof"].get("error_budget"))
        log.info("pass-2 calibration: ECE raw %.4f -> isotonic %.4f",
                 report["calibration"]["pass2_oof_raw"].get("ece", float("nan")),
                 report["calibration"]["pass2_oof_isotonic"].get("ece", float("nan")))

    # ---------------- 7. test decoding
    with T.stage("decode_test"):
        s1_ids_te, pool_ids_te = n1_te["id"].to_numpy(object), np_te["id"].to_numpy(object)
        k_t, IDX_t, P_t = decode(s1_te, P_te["cand"].to_numpy(), R["p2_te"], R["h_te"], len(n1_te), cfg["decoder"],
                                 R["fitted"])
        cand_col_te = P_te["cand"].to_numpy()
        sel_t = selections(k_t, IDX_t)
        pred = {s1_ids_te[i]: pool_ids_te[cand_col_te[rows]].tolist() for i, rows in enumerate(sel_t)}
        cand = _cand_map(P_te, n1_te, np_te)
    report["test_dashboard"] = _dashboard(n1_te, k_t, P_t, has_te, n1_tr, k_o, P_o, has_tr)
    for c, v in report["test_dashboard"]["test"].items():
        o = report["test_dashboard"]["oof_train"].get(c)
        log.info("dashboard %-10s S1=%6d  no-cand %.3f  pred-empty %.3f  mean matches %.3f%s", c[:10], v["n_s1"],
                 v["no_candidates_rate"], v["pred_empty_rate"], v["mean_matches"],
                 f"  (OOF: pred-empty {o['pred_empty_rate']:.3f}, mean {o['mean_matches']:.3f})" if o else
                 "  (unseen in train)")
    if test_gold is not None:
        tg = {s: test_gold.get(s, []) for s in n1_te["id"]}
        ctry = dict(zip(n1_te["id"], n1_te["cty"]))
        report["heldout"] = evaluate(pred, tg, ctry, cand)
        report["heldout_blocking"] = blocking_kpis(cand, tg, len(np_te), ctry)
        log.info("held-out macro F0.5 = %.5f (PC %.4f)", report["heldout"]["macro_f05"],
                 report["heldout_blocking"]["PC"])
    if out_dir is not None:
        with T.stage("analysis_artifacts"):
            try:
                _write_artifacts(Path(out_dir), report, R, imp_all, P_tr, X_tr, y, p1_oof, sel_o, fold_pair,
                                 n1_tr, np_tr, pred_oof, gold_id, cf, h_label, P_te, X_te, p1_te, sel_t, n1_te, np_te)
            except Exception as e:  # diagnostics must never break a run
                log.warning("could not write analysis artifacts: %r", e)
    report["checkpoints"] = ck.summary()
    return pred, cand


def _meta_stage(U_tr, U_te, n1_tr, np_tr, n1_te, np_te, gkeys, n2_tr, fold_s1, gold_id, country_tr, cfg, backend,
                seeds, ck, k_meta):
    """Learned meta-blocking (the last filter before the matcher) + blocking KPIs."""
    mc = cfg["meta"]
    yU = _labels(U_tr, gkeys, n2_tr)
    info = None
    if mc.get("enabled", True) and len(U_tr) and yU.any() and not yU.all():
        Xm_tr = U_tr[B.META_FEATS].to_numpy(np.float32)
        Xm_te = U_te[B.META_FEATS].to_numpy(np.float32)
        # a filter model needs few easy negatives: fit on all positives + a sample of negatives (scores for all
        # rows stay out-of-fold, and tau is chosen on these same scores, so the shift in scale is harmless)
        rate = float(mc.get("neg_sample", 1.0))
        mask = None
        if rate < 1.0 and len(yU) > int(mc.get("neg_sample_min_rows", 300000)):
            mask = yU | (np.random.RandomState(cfg["seed"]).rand(len(yU)) < rate)
            log.info("meta-blocker: training on all %d positives + %.0f%% of negatives (%d of %d rows)",
                     int(yU.sum()), 100 * rate, int(mask.sum()), len(yU))
        meta_oof, meta_te, info = fit_oof(Xm_tr, yU, fold_s1[U_tr["s1"].to_numpy()], Xm_te, B.META_FEATS, cfg,
                                          backend, seeds[:1], name="meta_blocker", params=_small_params(cfg, backend),
                                          rounds=1500, es=50, ckpt=ck, ckpt_key=k_meta + "-gbdt", train_mask=mask)
        del Xm_tr, Xm_te
        tau = B.choose_tau(U_tr["s1"].to_numpy(), meta_oof, yU, mc["top_keep"], mc["cap"], mc["recall_ratio"])
        keep_tr = B.select_candidates(U_tr["s1"].to_numpy(), meta_oof, mc["top_keep"], tau, mc["cap"])
        keep_te = B.select_candidates(U_te["s1"].to_numpy(), meta_te, mc["top_keep"], tau, mc["cap"])
        info = {**{k: v for k, v in info.items() if not k.startswith("_")}, "tau": tau}
        log.info("meta-blocker: tau=%.4f keeps %d/%d train and %d/%d test pairs", tau, int(keep_tr.sum()), len(U_tr),
                 int(keep_te.sum()), len(U_te))
    else:
        meta_oof = U_tr["n_ch"].to_numpy(np.float64) + U_tr["j_cos_c"].to_numpy()
        meta_te = U_te["n_ch"].to_numpy(np.float64) + U_te["j_cos_c"].to_numpy()
        keep_tr, keep_te = np.ones(len(U_tr), bool), np.ones(len(U_te), bool)
    U_tr["meta"], U_te["meta"] = meta_oof, meta_te
    P_tr = U_tr[keep_tr].reset_index(drop=True)
    P_te = U_te[keep_te].reset_index(drop=True)
    y = yU[keep_tr].astype(np.float32)
    cu = _cand_map(U_tr, n1_tr, np_tr)
    cf = _cand_map(P_tr, n1_tr, np_tr)
    uniq = {}
    for ch in B.CHANNELS:
        hit = U_tr[f"s_{ch}"].to_numpy() > 0
        only = hit & (U_tr["n_ch"].to_numpy() == 1)
        uniq[ch] = {"true_pairs_found": int((yU & hit).sum()), "unique_true_pairs": int((yU & only).sum())}
    blocking = {
        "train_union": blocking_kpis(cu, gold_id, n2_tr, country_tr, uniq),
        "train_final": blocking_kpis(cf, gold_id, n2_tr, country_tr),
        "test_final": {"pairs": int(len(P_te)), "cand_per_s1_mean": len(P_te) / max(len(n1_te), 1),
                       "RR": 1 - len(P_te) / max(len(n1_te) * len(np_te), 1), "union_pairs": int(len(U_te)),
                       "s1_without_candidates": int(len(n1_te) - P_te["s1"].nunique())},
    }
    for ch, v in uniq.items():
        log.info("  channel %s: %6d true pairs found, %5d found by no other channel", ch, v["true_pairs_found"],
                 v["unique_true_pairs"])
    return {"P_tr": P_tr, "P_te": P_te, "y": y, "cf": cf, "blocking": blocking, "meta_info": info}


def _write_artifacts(out, report, R, imp_all, P_tr, X_tr, y, p1_oof, sel_o, fold_pair, n1_tr, np_tr, pred_oof,
                     gold_id, cf, h_label, P_te, X_te, p1_te, sel_t, n1_te, np_te):
    """Tables for offline analysis: every OOF / test pair with probabilities and key features, per-entity
    results, full feature importances, the decoder grid and the worst errors per bucket."""
    out.mkdir(parents=True, exist_ok=True)
    files = {}

    def pairs_table(P, X, p1, p2, p2cal, sel_rows, n1, npool, extra):
        sel = np.zeros(len(P), bool)
        for rows in sel_rows:
            sel[rows] = True
        d = pd.DataFrame({"s1_id": n1["id"].to_numpy(object)[P["s1"].to_numpy()],
                          "cand_id": npool["id"].to_numpy(object)[P["cand"].to_numpy()],
                          "country": n1["cty"].to_numpy(object)[P["s1"].to_numpy()],
                          "p1": p1, "p2": p2, "p2_cal": p2cal, "selected": sel, **extra})
        for c in KEY_FEATS:
            if c in X:
                d[c] = X[c].to_numpy()
        return d

    iso = R["fitted"]["iso"]
    d = pairs_table(P_tr, X_tr, p1_oof, R["p2_oof"], iso.predict(R["p2_oof"]) if len(P_tr) else [], sel_o, n1_tr,
                    np_tr, {"y": y.astype(np.int8), "fold": fold_pair})
    files["oof_pairs"] = str(write_table(d, out / "oof_pairs"))
    d = pairs_table(P_te, X_te, p1_te, R["p2_te"], iso.predict(R["p2_te"]) if len(P_te) else [], sel_t, n1_te,
                    np_te, {})
    files["test_pairs"] = str(write_table(d, out / "test_pairs"))
    ids = n1_tr["id"].to_numpy(object)
    ent = pd.DataFrame({"s1_id": ids, "country": n1_tr["cty"].to_numpy(object),
                        "n_gold": [len(gold_id[s]) for s in ids],
                        "n_cands": [len(cf.get(s, ())) for s in ids],
                        "n_gold_in_cands": [len(set(gold_id[s]) & set(cf.get(s, ()))) for s in ids],
                        "n_pred": [len(pred_oof.get(s, ())) for s in ids],
                        "tp": [len(set(pred_oof.get(s, ())) & set(gold_id[s])) for s in ids],
                        "f05": [f05_entity(pred_oof.get(s, ()), gold_id[s]) for s in ids],
                        "h_oof": R["h_oof"], "has_true_candidate": h_label})
    files["oof_entities"] = str(write_table(ent, out / "oof_entities"))
    fi = pd.DataFrame({"feature": sorted(set(imp_all["pass1"]) | set(imp_all["pass2"]))})
    fi["pass1_gain"] = fi["feature"].map(imp_all["pass1"]).fillna(0.0)
    fi["pass2_gain"] = fi["feature"].map(imp_all["pass2"]).fillna(0.0)
    fi.sort_values("pass2_gain", ascending=False).to_csv(out / "feature_importance.csv", index=False)
    files["feature_importance"] = str(out / "feature_importance.csv")
    pd.DataFrame([{"oof_f05_in_sample": f, "decoder": c[0], "gamma": c[1], "p1": c[2], "p2": c[3], "p3": c[4]}
                  for f, c in R["fitted"].get("grid", [])]).to_csv(out / "decoder_grid.csv", index=False)
    files["decoder_grid"] = str(out / "decoder_grid.csv")
    _error_samples(out / "oof_errors.tsv", pred_oof, gold_id, cf, n1_tr, np_tr)
    files["oof_errors"] = str(out / "oof_errors.tsv")
    report["artifacts"] = files
    log.info("analysis artifacts: %s", ", ".join(Path(p).name for p in files.values()))


def _dashboard(n1_te, k_t, P_t, has_te, n1_tr, k_o, P_o, has_tr):
    """Per-country sanity numbers on test vs OOF: predicted-empty rate, mean matches, top-1 prob histogram."""
    def one(n1, k, P, has):
        c = n1["cty"].to_numpy(object)
        out = {}
        for key in sorted(set(c)):
            m = c == key
            top = P[m, 0] if len(P) else np.zeros(0)
            out[key] = {"n_s1": int(m.sum()), "no_candidates_rate": float((~has[m]).mean()),
                        "pred_empty_rate": float((k[m] == 0).mean()), "mean_matches": float(k[m].mean()),
                        "top1_prob_hist": np.histogram(top, bins=10, range=(0, 1))[0].tolist()}
        return out
    return {"test": one(n1_te, k_t, P_t, has_te), "oof_train": one(n1_tr, k_o, P_o, has_tr)}


__all__ = ["fit_predict", "COLLECTIVE"]
