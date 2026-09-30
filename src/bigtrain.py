"""Bigger pair model: 2x the training S1s, larger LightGBM, capacity focused on hard negatives.

Why: the leaderboard confirmed that test holds ~2x the look-alike records of train (shift-corrected decoding +0.0044),
and the holdout error tree puts the remaining loss in the model's gray zone (missed extra matches, look-alike false
positives). So the pass-1 / pass-2 models are refit with
  * the original 200k training S1s (their cached candidates + features) plus `extra` S1s drawn from train S1s that
    are neither in that sample nor in the holdout (src/proxy.py), blocked and scored exactly like test (all S1s in
    the index, meta-blocker = mean of its fold models) - so the holdout stays an honest judge;
  * a larger LightGBM (more leaves, learning rate 0.05 instead of the 0.1 shortcut, more rounds);
  * a training mask: all positives, every S1's top-`hard_rank` candidates by meta score (where look-alikes live) and
    `easy_rate` of the rest. OOF predictions are still made for every pair (pass-2 needs them).
The meta-blocker, blocking, features and within-S1 collective features are unchanged, so the test path is
scale.score_chunk with the new fold models.

    python -m src.bigtrain train   [--extra 200000]
    python -m src.bigtrain proxy   (re-score the holdout with the new models)
    python -m src.bigtrain test    (re-score every test S1 with the new models)
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import blocking as B
from .bigblock import PartitionIndex
from .cache import Checkpointer, code_hash
from .collective import WITHIN, sibling_edges, within_collective
from .features import monotone_vector, partition_stats
from .io_utils import load_gold
from .models import fit_oof, make_folds, predict_models
from .proxy import proxy_key, run_keys
from .scale import (_cand_strings, _chunks, _features_batched, load_fold_models, load_partition, normalized_source,
                    partition_keys, score_chunk)
from .track import rss_gb

log = logging.getLogger("ber")
BIG_LGB = {"learning_rate": 0.05, "num_leaves": 127, "min_data_in_leaf": 100, "feature_fraction": 0.7,
           "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 2.0}
BIG_ROUNDS, BIG_ES = 8000, 100


def _ctx(a):
    run_rep = json.loads((Path(a.report) / "run_report.json").read_text(encoding="utf-8"))
    cfg = run_rep["config"]
    d = Path(a.data)
    paths = {sp: {k: normalized_source(d / sp / f"{sp}_source{k}.tsv", Path(a.cache)) for k in (1, 2, 3)}
             for sp in ("train", "test")}
    ck = Checkpointer(a.cache)
    tau = float(run_rep["meta_blocker"]["tau"])
    bcfg = dict(cfg["blocking"])
    bcfg.update(cfg["scale"].get("blocking", {}))
    return run_rep, cfg, paths, ck, tau, bcfg


def manifest_path(a):
    return Path(a.report) / "bigtrain" / "manifest.json"


# ------------------------------------------------------------------ training data
def extra_pairs(a, cfg, paths, ck, tau, bcfg, base_ids, holdout_ids, meta_models, feats):
    """Candidates + features + sibling edges for `extra` train S1s outside the base sample and the holdout,
    blocked against the full partition exactly like test. Returns per-pair arrays and the extra S1 ids."""
    s1_meta = pq.read_table(paths["train"][1], columns=["id", "cty"]).to_pandas()
    excluded = set(base_ids) | set(holdout_ids)
    pool_ids = s1_meta["id"].to_numpy(object)
    ok = np.fromiter((i not in excluded for i in pool_ids), bool, len(pool_ids))
    rng = np.random.RandomState(101)
    pick = np.sort(rng.choice(np.flatnonzero(ok), min(a.extra, int(ok.sum())), replace=False))
    ext_ids = pool_ids[pick]
    ext_code = {s: i for i, s in enumerate(ext_ids)}
    gold_all = load_gold(Path(a.data) / "train" / "train_ground_truth.tsv")
    gold = {s: gold_all.get(s, []) for s in ext_ids}              # only the extra S1s (memory)
    del gold_all, pool_ids, ok, excluded
    gc.collect()
    mc, top_m = cfg["meta"], int(cfg.get("collective", {}).get("top_m", 10))
    chunk_s1 = int(cfg["scale"].get("chunk_s1", 100000))
    key = ck.key("big-extra", a.extra, 101, len(base_ids), len(holdout_ids), bcfg, float(tau), mc, top_m,
                 code_hash("bigblock", "features", "collective"), "v1")
    out = []
    for part in partition_keys([paths["train"][1]]):
        t0 = time.time()
        s1 = load_partition([paths["train"][1]], part)
        pool = load_partition([paths["train"][2], paths["train"][3]], part)
        ids1 = s1["id"].to_numpy(object)
        rows = np.flatnonzero(np.fromiter((i in ext_code for i in ids1), bool, len(ids1)))
        idsp = pd.Index(pool["id"].to_numpy(object))
        gl = [gold.get(i, []) for i in ids1[rows]]
        gpos = idsp.get_indexer([c for g in gl for c in g])
        gs = np.repeat(rows, [len(g) for g in gl])
        gkey = np.unique(gs[gpos >= 0].astype(np.int64) * len(pool) + gpos[gpos >= 0])
        idx = stats = None
        for ci, ch in enumerate(_chunks(rows, chunk_s1)):
            kc = f"{key}-{part}-chunk{ci}"
            r = ck.load(kc)
            if r is None:
                if idx is None:
                    idx = PartitionIndex(s1, pool, bcfg)
                    stats = partition_stats(s1, pool)
                    log.info("  extra %s: index built (%.0fs, RSS %s GB)", part, time.time() - t0, rss_gb())
                u = idx.query(ch)
                m = predict_models(meta_models, u[B.META_FEATS].to_numpy(np.float32))
                kp = B.select_candidates(u["s1"].to_numpy(), m, mc["top_keep"], tau, mc["cap"])
                Pc = u[kp].reset_index(drop=True)
                Pc["meta"] = m[kp].astype(np.float32)
                del u, m
                X = _features_batched(Pc, s1, pool, idx.idf, stats, part)[feats].to_numpy(np.float32)
                sl, cd = Pc["s1"].to_numpy(np.int64), Pc["cand"].to_numpy(np.int64)
                nm, ad = _cand_strings(pool, cd)
                I, J, sim, sn = sibling_edges(sl, Pc["meta"].to_numpy(), nm, ad, top_m)
                code = np.array([ext_code[i] for i in ids1[sl]], np.int64)
                r = {"s1": code, "cand": cd, "part": part, "meta": Pc["meta"].to_numpy(np.float32),
                     "y": np.isin(sl * len(pool) + cd, gkey), "X": X, "E": (I, J, sim, sn)}
                ck.save(kc, r)
                del Pc, X
            out.append(kc)                                    # finished chunks wait on disk, not in memory
            log.info("  extra %s chunk %d: %d S1 -> %d pairs, %d true | %.0fs, RSS %s GB", part, ci + 1, len(ch),
                     len(r["s1"]), int(r["y"].sum()), time.time() - t0, rss_gb())
            del r
            gc.collect()
        del idx, stats, s1, pool
        gc.collect()
    out = [ck.load(kc) for kc in out]
    G_ext = np.array([len(gold.get(s, [])) for s in ext_ids])
    cty_ext = s1_meta.set_index("id")["cty"].reindex(ext_ids).to_numpy(object)
    return out, ext_ids, G_ext, cty_ext


def train(a):
    run_rep, cfg, paths, ck, tau, bcfg = _ctx(a)
    sc = cfg["scale"]
    k_meta, _, _ = run_keys(cfg, paths, ck, int(sc["train_s1"]), tau)
    meta_models = load_fold_models(ck, k_meta, cfg["n_folds"])
    # base sample: same draw as fit_predict_scale, its cached candidates / features / sibling edges
    s1_meta = pq.read_table(paths["train"][1], columns=["id", "cty"]).to_pandas()
    n_take = min(int(sc["train_s1"]), len(s1_meta))
    take = np.sort(np.random.RandomState(cfg["seed"]).choice(len(s1_meta), n_take, replace=False))
    base_ids = np.array(sorted(s1_meta["id"].to_numpy(object)[take]), dtype=object)
    k_feat = ck.key("scale-feat", k_meta, float(tau), code_hash("features", "normalize", "bigblock"))
    FX = ck.load(k_feat)
    if FX is None:
        raise SystemExit(f"base features not cached ({k_feat})")
    feats = list(FX[0].columns)                                # the pair-feature order of the base run
    del FX                                                     # re-loaded after the extra pairs: the partition
    gc.collect()                                               # indexes need that memory
    holdout = ck.load(proxy_key(a.n))
    if holdout is None:                                        # fresh run: same deterministic selection, no models
        from .proxy import build_proxy
        holdout = build_proxy(a.data, cfg, ck, tau, a.n, select_only=True)
        log.info("holdout ids selected (no scoring): %d", len(holdout["ids"]))
    holdout_ids = holdout["ids"]
    del holdout
    gc.collect()
    ext, ext_ids, G_ext, cty_ext = extra_pairs(a, cfg, paths, ck, tau, bcfg, base_ids, holdout_ids, meta_models,
                                               feats)
    X_b, P_b, E_b = ck.load(k_feat)
    Xb = X_b[feats].to_numpy(np.float32)
    del X_b
    gold = load_gold(Path(a.data) / "train" / "train_ground_truth.tsv")
    G_b = np.array([len(gold.get(s, [])) for s in base_ids])
    cty_b = s1_meta.set_index("id")["cty"].reindex(base_ids).to_numpy(object)
    del gold, s1_meta
    nb = len(base_ids)
    # ---- stack base + extra (pairs keep their row order; extra S1 codes follow the base codes)
    X = np.concatenate([Xb] + [r["X"] for r in ext])
    del Xb
    for r in ext:
        r["X"] = None
    y = np.concatenate([P_b["y"].to_numpy()] + [r["y"] for r in ext]).astype(np.float32)
    s1 = np.concatenate([P_b["s1"].to_numpy(np.int64)] + [r["s1"] + nb for r in ext])
    meta = np.concatenate([P_b["meta"].to_numpy(np.float32)] + [r["meta"] for r in ext])
    E, off = [list(E_b)], len(P_b)
    for r in ext:
        I, J, sim, sn = r["E"]
        E.append([I + off, J + off, sim, sn])
        off += len(r["s1"])
    E = tuple(np.concatenate([e[i] for e in E]) for i in range(4))
    del P_b, ext
    G = np.r_[G_b, G_ext]
    cty = np.r_[cty_b, cty_ext]
    n_s1 = len(G)
    strata = np.array([f"{c}_{min(g, 3)}" for c, g in zip(cty, G)], dtype=object)
    fold_s1 = make_folds(strata, cfg["n_folds"], cfg["seed"])
    fold_pair = fold_s1[s1]
    # hard-negative focus: all positives, each S1's top candidates by meta score, a share of the rest
    rk = B.group_rank(s1, meta.astype(np.float64))
    keep = (y > 0) | (rk <= a.hard_rank) | (np.random.RandomState(7).rand(len(y)) < a.easy_rate)
    log.info("big training set: %d S1 (%d base + %d extra), %d pairs, %d true, train mask %.1f%% (RSS %s GB)",
             n_s1, nb, n_s1 - nb, len(y), int(y.sum()), 100 * keep.mean(), rss_gb())
    bcfg_m = dict(cfg, model=dict(cfg["model"], lgb=dict(BIG_LGB), num_boost_round=BIG_ROUNDS, early_stopping=BIG_ES))
    mono = monotone_vector(feats) if cfg["model"].get("monotone", True) else None
    seeds = cfg["seeds_cpu"]
    k_data = ck.key("big-data", k_feat, a.extra, a.n, a.hard_rank, a.easy_rate, "v1")
    k_b1 = ck.key("big-pass1", k_data, BIG_LGB, BIG_ROUNDS, BIG_ES, list(seeds))
    p1, _, info1 = fit_oof(X, y, fold_pair, None, feats, bcfg_m, "lightgbm", seeds, mono, name="big-pass1",
                           ckpt=ck, ckpt_key=k_b1, train_mask=keep, keep_models=True)
    info1.pop("_models")
    info1.pop("_importance_all", None)
    C = within_collective(s1, p1, E).to_numpy(np.float32)
    X = np.hstack([X, C])
    del C
    k_b2 = ck.key("big-pass2", k_b1, "within", code_hash("collective"))
    p2, _, info2 = fit_oof(X, y, fold_pair, None, feats + WITHIN, bcfg_m, "lightgbm", seeds,
                           (mono + [0] * len(WITHIN)) if mono else None, name="big-pass2", ckpt=ck, ckpt_key=k_b2,
                           train_mask=keep, keep_models=True)
    info2.pop("_models")
    info2.pop("_importance_all", None)
    # quick OOF check with the submitted decoder shape (in-sample calibration; the holdout is the real judge)
    from .decide import f05_per_s1, run_policy
    from .models import fit_isotonic
    cand = np.zeros(len(s1), np.int64)                      # candidates of different S1s never collide here
    cand[:] = np.arange(len(s1))
    for name, p in (("pass1", p1), ("pass2", p2)):
        iso = fit_isotonic(p, y)
        rows, _ = run_policy(s1, cand, p, iso, "hard", 1.25, 0.0, None, n_s1)
        f = f05_per_s1(rows, s1, y > 0, G, n_s1)
        log.info("big %s OOF macro F0.5 (train S1s, no exclusivity) %.5f | base part %.5f", name, f.mean(),
                 f[:nb].mean())
    man = {"meta": k_meta, "pass1": k_b1, "pass2": k_b2, "n_folds": cfg["n_folds"], "feats": feats,
           "n_train_s1": int(n_s1), "n_pairs": int(len(y)), "params": BIG_LGB, "hard_rank": a.hard_rank,
           "easy_rate": a.easy_rate, "pass1": k_b1, "info_pass1": info1, "info_pass2": info2}
    manifest_path(a).parent.mkdir(parents=True, exist_ok=True)
    manifest_path(a).write_text(json.dumps(man, indent=1, default=str), encoding="utf-8")
    log.info("manifest -> %s", manifest_path(a))


# ------------------------------------------------------------------ scoring with the new models
def load_models(a, ck):
    man = json.loads(manifest_path(a).read_text(encoding="utf-8"))
    models = tuple(load_fold_models(ck, man[k], man["n_folds"]) for k in ("meta", "pass1", "pass2"))
    return man, models


def score_test(a):
    """Every test S1 through score_chunk with the new fold models; one checkpoint per partition (+ per chunk)."""
    run_rep, cfg, paths, ck, tau, bcfg = _ctx(a)
    man, models = load_models(a, ck)
    mc, top_m = cfg["meta"], int(cfg.get("collective", {}).get("top_m", 10))
    chunk_s1 = int(cfg["scale"].get("chunk_s1", 100000))
    keys = {}
    for part in partition_keys([paths["test"][1]]):
        k_part = ck.key("big-test", part, str(paths["test"][1]), man["meta"], man["pass1"], man["pass2"], bcfg,
                        float(tau), mc, top_m, code_hash("bigblock", "features", "collective"))
        keys[part] = k_part
        if ck.load(k_part) is not None:
            continue
        t0 = time.time()
        s1 = load_partition([paths["test"][1]], part)
        pool = load_partition([paths["test"][2], paths["test"][3]], part)
        idx = stats = None
        acc = []
        chs = _chunks(np.arange(len(s1)), chunk_s1)
        for ci, ch in enumerate(chs):
            kc = f"{k_part}-chunk{ci}"
            r = ck.load(kc)
            if r is None:
                if idx is None:
                    idx = PartitionIndex(s1, pool, bcfg)
                    stats = partition_stats(s1, pool)
                r = score_chunk(idx, ch, s1, pool, stats, part, models, man["feats"], tau, mc, top_m)
                if r is None:
                    continue
                ck.save(kc, r)
            acc.append(r)
            el = time.time() - t0
            log.info("  big test %s chunk %d/%d: %d candidates | %.0fs, ETA %.0fs, RSS %s GB", part, ci + 1, len(chs),
                     len(r[0]), el, el / (ci + 1) * (len(chs) - ci - 1), rss_gb())
        ck.save(k_part, tuple(np.concatenate([x[i] for x in acc]) for i in range(5)))
        del idx, stats, s1, pool
    (Path(a.report) / "bigtrain" / "test_keys.json").write_text(json.dumps(keys, indent=1), encoding="utf-8")
    log.info("big test keys: %s", keys)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[3]
    ap.add_argument("stage", choices=["train", "proxy", "test"])
    ap.add_argument("--data", default=str(root / "student_resource" / "dataset"))
    ap.add_argument("--cache", default=str(root / "ber_cache"))
    ap.add_argument("--report", default=str(Path(__file__).resolve().parents[1] / "reports"))
    ap.add_argument("--n", type=int, default=150000, help="holdout size per country (src.proxy)")
    ap.add_argument("--extra", type=int, default=200000, help="extra training S1s")
    ap.add_argument("--hard-rank", dest="hard_rank", type=int, default=12)
    ap.add_argument("--easy-rate", dest="easy_rate", type=float, default=0.35)
    a = ap.parse_args(argv)
    from .run import setup_logging
    setup_logging(Path(a.report) / "bigtrain" / a.stage)
    t0 = time.time()
    if a.stage == "train":
        train(a)
    elif a.stage == "proxy":
        from .proxy import build_proxy
        run_rep, cfg, paths, ck, tau, bcfg = _ctx(a)
        man, models = load_models(a, ck)
        P = build_proxy(a.data, cfg, ck, tau, a.n, models=(man, models))
        ck.save(proxy_key(a.n, tag=man["pass2"]), P)
        log.info("big proxy saved: %d S1, %d pairs", len(P["ids"]), len(P["s"]))
    else:
        score_test(a)
    log.info("stage %s done in %.0fs", a.stage, time.time() - t0)


if __name__ == "__main__":
    main()
