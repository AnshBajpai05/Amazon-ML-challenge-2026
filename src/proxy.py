"""Test-like holdout ("proxy") built from train and scored by the exact test path, for the decision layer.

The decoder of the first submission was tuned on OOF predictions of a random 9% train sample (nested OOF
F0.5 0.967, public LB 0.950). That OOF differs from test in three ways that matter to decisions, not to the
pair model:
  * density    test has ~5.8 pool records per S1 vs 4.67 in train, i.e. about twice the unmatched records;
  * conflicts  a 9% sample rarely holds two S1s competing for one pool record, test holds every S1
               (claim conflicts: OOF 0.03% vs test 0.24-1.17%);
  * averaging  OOF has one fold model per pair, test the mean of the 5 fold models (flatter probabilities).
The proxy removes all three: whole cities of train S1s that were NOT in the training sample; extra S1s removed at
random until pool/S1 matches test (their pool records become unmatched records, exactly like test ones); a
blocking index over the remaining S1s only; and the saved fold models averaged as on test (scale.score_chunk).
Labels come from train gold, so every decision-layer change can be measured before it is submitted.

    python -m src.proxy --n 150000        # builds (checkpointed) and prints the proxy KPIs
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .bigblock import PartitionIndex
from .blocking import group_rank
from .cache import Checkpointer, code_hash
from .features import partition_stats
from .io_utils import load_gold
from .models import detect_backend
from .scale import _chunks, _small, load_fold_models, load_partition, normalized_source, score_chunk
from .track import rss_gb

log = logging.getLogger("ber")
VERSION = "v1"                 # bump when the holdout construction changes (the scoring is checkpointed)


def region_key(raw_addr):
    """City-level region from the raw address: the second-to-last comma field ('..., Pune, Maharashtra',
    '..., Penfield, NY'); records without commas fall back to their last field."""
    parts = [p.strip().lower() for p in str(raw_addr).split(",") if p.strip()]
    if not parts:
        return ""
    return parts[-2] if len(parts) >= 2 else parts[-1]


def run_keys(cfg, paths, ck, n_train, tau):
    """Checkpoint keys of the fitted models; mirrors the key chain of scale.fit_predict_scale."""
    bcfg = dict(cfg["blocking"])
    bcfg.update(cfg.get("scale", {}).get("blocking", {}))
    backend = detect_backend(cfg.get("backend", "auto"))
    seeds = cfg["seeds_gpu"] if backend == "xgboost_cuda" else cfg["seeds_cpu"]
    seed, mc = cfg["seed"], cfg["meta"]
    top_m = int(cfg.get("collective", {}).get("top_m", 10))
    k_union = ck.key("scale-union", {s: str(p) for s, p in paths["train"].items()}, bcfg, n_train, seed,
                     code_hash("bigblock", "blocking"))
    k_meta = ck.key("scale-meta", k_union, mc, _small(cfg, backend), backend, seed, code_hash("models"))
    k_feat = ck.key("scale-feat", k_meta, float(tau), code_hash("features", "normalize", "bigblock"))
    k_p1 = ck.key("scale-pass1", k_feat, cfg["model"], backend, list(seeds), code_hash("models"))
    k_p2 = ck.key("scale-pass2", k_p1, top_m, code_hash("models", "collective"))
    return k_meta, k_p1, k_p2


def build_proxy(data_dir, cfg, ck, tau, n_per_country=150000, seed=7, models=None, select_only=False):
    """Returns a dict of flat arrays over proxy pairs (s, c, p2, meta, p1, y) and proxy S1s (ids, cty, region,
    G = gold count, incl. gold records the blocking missed), plus blocking-recall counts beyond the cap.
    models: optional (manifest, (meta, pass1, pass2) fold models) to score the same holdout with other models.
    select_only: return just {"ids": holdout S1 ids} (deterministic selection, no models needed)."""
    sc = cfg.get("scale", {})
    bcfg = dict(cfg["blocking"])
    bcfg.update(sc.get("blocking", {}))
    mc = cfg["meta"]
    top_m = int(cfg.get("collective", {}).get("top_m", 10))
    chunk_s1 = int(sc.get("chunk_s1", 100000))
    d = Path(data_dir)
    paths = {split: {s: normalized_source(d / split / f"{split}_source{s}.tsv", Path(ck.root)) for s in (1, 2, 3)}
             for split in ("train", "test")}

    # the training sample of fit_predict_scale (same RNG draw), never part of the proxy
    s1_meta = pq.read_table(paths["train"][1], columns=["id", "cty"]).to_pandas()
    n_take = min(int(sc.get("train_s1", 300000)), len(s1_meta))
    take = np.sort(np.random.RandomState(cfg["seed"]).choice(len(s1_meta), n_take, replace=False))
    in_train = np.zeros(len(s1_meta), bool)
    in_train[take] = True
    trained = set(s1_meta["id"].to_numpy(object)[in_train])
    if select_only:
        k_meta = k_p1 = k_p2 = feats = None
    elif models is None:
        k_meta, k_p1, k_p2 = run_keys(cfg, paths, ck, n_take, tau)
        models = tuple(load_fold_models(ck, k, cfg["n_folds"]) for k in (k_meta, k_p1, k_p2))
    else:
        man, models = models
        k_meta, k_p1, k_p2 = man["meta"], man["pass1"], man["pass2"]
    if not select_only:
        feats = models[1][0][1][0][1].feature_name()             # pass-1 feature order, as trained
        log.info("proxy: models %s / %s / %s, %d pass-1 features", k_meta, k_p1, k_p2, len(feats))

    # pool/S1 density per country: train vs test (label-free)
    def density(split):
        s1 = pq.read_table(paths[split][1], columns=["cty"]).to_pandas()["cty"].value_counts()
        pool = pd.concat([pq.read_table(paths[split][k], columns=["cty"]).to_pandas()["cty"] for k in (2, 3)])
        return pool.value_counts() / s1
    dens_tr, dens_te = density("train"), density("test")

    gold = load_gold(d / "train" / "train_ground_truth.tsv")
    rng = np.random.RandomState(seed)
    out = {k: [] for k in ("s", "c", "p2", "meta", "p1", "y")}
    s1_ids, s1_cty, s1_reg, s1_G = [], [], [], []
    recall_rows, off_s, off_c, info = [], 0, 0, {}
    for part in sorted(set(dens_tr.index) & set(dens_te.index)):
        t0 = time.time()
        s1 = load_partition([paths["train"][1]], part)
        pool = load_partition([paths["train"][2], paths["train"][3]], part)
        ids = s1["id"].to_numpy(object)
        tr = np.fromiter((i in trained for i in ids), bool, len(ids))
        # remove S1s until pool/S1 matches test: the training sample plus a random extra share
        f_total = float(np.clip(1.0 - dens_tr[part] / dens_te[part], tr.mean(), 0.6))
        extra = (f_total - tr.mean()) / max(1.0 - tr.mean(), 1e-9)
        present = ~tr & (rng.rand(len(ids)) >= extra)
        s1p = s1[present].reset_index(drop=True)
        idsp = ids[present]
        reg = np.array([region_key(a) for a in s1p["raw_addr"].to_numpy(object)], dtype=object)
        # whole regions (cities) in random order until the budget is reached
        vc = pd.Series(reg).value_counts()
        order = vc.index.to_numpy(object)[rng.permutation(len(vc))]
        cum = np.cumsum(vc.reindex(order).to_numpy())
        chosen = set(order[:int(np.searchsorted(cum, n_per_country)) + 1])
        rows = np.flatnonzero(np.fromiter((r in chosen for r in reg), bool, len(reg)))
        info[part] = {"s1_partition": int(len(ids)), "removed_share": float(1 - present.mean()),
                      "pool_per_present_s1": float(len(pool) / max(present.sum(), 1)),
                      "test_pool_per_s1": float(dens_te[part]), "proxy_s1": int(len(rows)),
                      "regions": int(len(chosen))}
        log.info("proxy %s: %d S1 (%.1f%% removed -> pool/S1 %.2f, test %.2f), %d proxy S1 in %d regions", part,
                 len(ids), 100 * (1 - present.mean()), info[part]["pool_per_present_s1"], dens_te[part], len(rows),
                 len(chosen))
        if select_only:
            s1_ids.append(idsp[rows])
            del s1, pool, s1p
            continue

        idx_pool = pd.Index(pool["id"].to_numpy(object))
        gl = [gold.get(i, []) for i in idsp[rows]]
        G = np.array([len(g) for g in gl])
        gs = np.repeat(rows, G)
        gpos = idx_pool.get_indexer([c for g in gl for c in g])
        gkey = np.unique(gs[gpos >= 0].astype(np.int64) * len(pool) + gpos[gpos >= 0])
        key = ck.key("proxy", VERSION, part, n_per_country, seed, k_meta, k_p1, k_p2, bcfg, float(tau), mc, top_m,
                     code_hash("bigblock", "features", "collective"))
        idx = stats = None
        acc = []
        for ci, ch in enumerate(_chunks(rows, chunk_s1)):
            kc = f"{key}-chunk{ci}"
            r = ck.load(kc)
            if r is None:
                if idx is None:
                    idx = PartitionIndex(s1p, pool, bcfg)
                    stats = partition_stats(s1p, pool)
                    log.info("    %s index over %d present S1 x %d pool built (%.0fs, RSS %s GB)", part, len(s1p),
                             len(pool), time.time() - t0, rss_gb())
                res = score_chunk(idx, ch, s1p, pool, stats, part, models, feats, tau, mc, top_m, union=True)
                if res is None:
                    continue
                us, uc, um = res[5]
                ut = np.isin(us * len(pool) + uc, gkey)
                rk = group_rank(us, um.astype(np.float64))            # meta rank of every union pair in its S1
                r = res[:5] + ((us[ut], rk[ut]),)                     # keep only the true union pairs' ranks
                ck.save(kc, r)
            acc.append(r)
            log.info("    %s proxy chunk %d: %d S1 -> %d candidates | %.0fs, RSS %s GB", part, ci + 1, len(ch),
                     len(r[0]), time.time() - t0, rss_gb())
        s, c, p2, meta, p1 = (np.concatenate([a[i] for a in acc]) for i in range(5))
        code = np.full(len(s1p), -1, np.int64)
        code[rows] = np.arange(len(rows)) + off_s
        y = np.isin(s.astype(np.int64) * len(pool) + c, gkey)
        out["s"].append(code[s])
        out["c"].append(c.astype(np.int64) + off_c)
        out["p2"].append(p2)
        out["meta"].append(meta)
        out["p1"].append(p1)
        out["y"].append(y)
        recall_rows.append(np.concatenate([a[5][1] for a in acc]))
        s1_ids.append(idsp[rows])
        s1_cty.append(np.full(len(rows), part, dtype=object))
        s1_reg.append(reg[rows])
        s1_G.append(G)
        info[part]["gold_pairs"] = int(G.sum())
        info[part]["gold_pairs_in_pool"] = int(len(gkey))
        off_s += len(rows)
        off_c += len(pool)
        del idx, stats, s1, pool, s1p
    if select_only:
        return {"ids": np.concatenate(s1_ids), "info": info}
    P = {k: np.concatenate(v) for k, v in out.items()}
    P.update(ids=np.concatenate(s1_ids), cty=np.concatenate(s1_cty), region=np.concatenate(s1_reg),
             G=np.concatenate(s1_G), true_union_meta_rank=np.concatenate(recall_rows), info=info)
    return P


def region_folds(P, n_folds=5, seed=11):
    """Fold per proxy S1 by region (a city's S1s and their conflicts stay in one fold)."""
    key = pd.Series(P["cty"]).astype(str) + "|" + pd.Series(P["region"]).astype(str)
    u, inv = np.unique(key.to_numpy(object), return_inverse=True)
    perm = np.random.RandomState(seed).permutation(len(u))
    return (perm % n_folds)[inv]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[3]
    ap.add_argument("--data", default=str(root / "student_resource" / "dataset"))
    ap.add_argument("--cache", default=str(root / "ber_cache"))
    ap.add_argument("--report", default=str(Path(__file__).resolve().parents[1] / "reports"))
    ap.add_argument("--n", type=int, default=150000, help="proxy S1s per country")
    a = ap.parse_args(argv)
    from .run import setup_logging
    setup_logging(Path(a.report) / "proxy")
    rep = json.loads((Path(a.report) / "run_report.json").read_text(encoding="utf-8"))
    cfg = rep["config"]                                   # the resolved config the submitted models were fit with
    ck = Checkpointer(a.cache)
    P = build_proxy(a.data, cfg, ck, rep["meta_blocker"]["tau"], a.n)
    ck.save(proxy_key(a.n), P)
    tu = P["true_union_meta_rank"]
    G = P["G"].sum()
    log.info("proxy built: %d S1, %d pairs, gold %d | recall union %.4f, @cap25 %.4f, @cap40 %.4f, candidates %.4f",
             len(P["ids"]), len(P["s"]), G, len(tu) / G, (tu <= 25).sum() / G, (tu <= 40).sum() / G,
             P["y"].sum() / G)
    log.info("proxy info: %s", json.dumps(P["info"]))


def proxy_key(n, tag=None):
    """Checkpoint of a built proxy; tag = pass-2 model key when scored with models other than the submitted ones."""
    return f"proxy-built-{VERSION}-{n}" + (f"-{tag}" if tag else "")


if __name__ == "__main__":
    main()
