"""Large-data pipeline: millions of records, bounded memory, resumable (used when the pool is big).

Flow (every step checkpointed; one country partition in memory at a time; ids handled as integer codes):
  0. read each TSV with pyarrow, normalize in parallel, cache as parquet (reused by every later run)
  1. train: a sample of S1 entities (default 300k) is blocked against the FULL train pool of its partition
     (realistic competition); reverse channels use every train S1 -> candidate union with labels
  2. meta-blocker (OOF on the union, negatives subsampled) -> tau -> train candidates
  3. pair features for the train candidates (chunked) -> pass-1 GBDT (OOF, fold models kept)
  4. decoder tuned by nested CV on the OOF probabilities (exclusivity + expected-F0.5 / threshold policies)
  5. test: per partition, S1 chunks: blocking -> meta score -> candidates -> features -> p1 (compact arrays)
  6. exclusivity + decoding over all test pairs -> matches; candidates = the meta-blocked set
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

from . import blocking as B
from .bigblock import PartitionIndex
from .cache import NULL, code_hash
from .decode import decode, selections, tune_decoder
from .collective import WITHIN, sibling_edges, within_collective
from .features import monotone_vector, pair_features, partition_stats
from .io_utils import load_gold
from .metrics import evaluate
from .models import binary_metrics, calibration_report, detect_backend, fit_oof, make_folds, predict_models
from .normalize import normalize_frame
from .track import Tracker, rss_gb, write_table

log = logging.getLogger("ber")
COLS = ["entity_id", "business_name", "business_address", "country"]


# ------------------------------------------------------------------ 0. normalized, cached sources
def _read_arrow(path):
    return pcsv.read_csv(path, read_options=pcsv.ReadOptions(block_size=1 << 26),
                         parse_options=pcsv.ParseOptions(delimiter="\t", quote_char=False, newlines_in_values=False,
                                                         invalid_row_handler=lambda r: "skip"),
                         convert_options=pcsv.ConvertOptions(column_types={c: pa.string() for c in COLS},
                                                             strings_can_be_null=False, include_columns=COLS))


def normalized_source(path, cache_dir, chunk=400000):
    """Normalized parquet for one source TSV, cached by file size/mtime + normalizer code."""
    st = os.stat(path)
    from .normalize import LEX_HASH
    key = f"norm-{Path(path).stem}-{st.st_size}-{int(st.st_mtime)}-{code_hash('normalize')}{LEX_HASH[:6]}"
    out = Path(cache_dir) / f"{key}.parquet"
    if out.exists():
        log.info("  normalized cache hit: %s", out.name)
        return out
    t0 = time.time()
    tbl = _read_arrow(path)
    tmp = out.with_suffix(".tmp")
    writer = None
    for s in range(0, tbl.num_rows, chunk):
        df = tbl.slice(s, chunk).to_pandas()
        df = df[df["entity_id"].str.strip() != ""].reset_index(drop=True)
        t = pa.Table.from_pandas(normalize_frame(df), preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(tmp, t.schema, compression="zstd")
        writer.write_table(t)
        done = min(s + chunk, tbl.num_rows)
        el = time.time() - t0
        log.info("    normalize %s: %d/%d rows (%.0fs, ETA %.0fs, RSS %s GB)", Path(path).name, done, tbl.num_rows,
                 el, el / done * (tbl.num_rows - done), rss_gb())
    writer.close()
    os.replace(tmp, out)
    return out


def load_partition(paths, cty):
    """Rows of the given normalized parquet files whose partition key is cty (pyarrow-backed strings)."""
    frames = [pd.read_parquet(p, filters=[("cty", "==", cty)], dtype_backend="pyarrow") for p in paths]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def partition_keys(paths):
    keys = set()
    for p in paths:
        keys |= set(pq.read_table(p, columns=["cty"]).column(0).unique().to_pylist())
    return sorted(keys)


def _chunks(a, n):
    return [a[i:i + n] for i in range(0, len(a), n)]


def _small(cfg, backend):
    return dict(cfg["model"]["lgb_small"] if backend == "lightgbm" else cfg["model"]["xgb_small"])


def _features_batched(P, s1, pool, idf, stats, part, batch=400000):
    out = []
    for i in range(0, len(P), batch):
        sub = P.iloc[i:i + batch].reset_index(drop=True)
        sub["part"] = part
        out.append(pair_features(sub, s1, pool, {part: idf}, stats=stats))
    return pd.concat(out, ignore_index=True) if out else None


def _cand_strings(pool, cand):
    """Candidate name (core, else full) and full address strings for sibling similarity."""
    core = pool["n_core"].iloc[cand].to_numpy(object)
    full = pool["n_full"].iloc[cand].to_numpy(object)
    names = np.where(pd.isna(core) | (core == ""), full, core)
    addrs = pool["a_full"].iloc[cand].to_numpy(object)
    return np.array([x if isinstance(x, str) else "" for x in names], dtype=object),         np.array([x if isinstance(x, str) else "" for x in addrs], dtype=object)


def kpis(s1c, y, n_s1, G, n_pool, country=None):
    """Vectorised blocking KPIs over integer-coded pairs (PC, entity completeness, RR, PQ, cand/S1)."""
    found = np.bincount(s1c, weights=y.astype(np.float64), minlength=n_s1)
    nc = np.bincount(s1c, minlength=n_s1)

    def one(m):
        g = G[m]
        return {"n_s1": int(m.sum()), "true_pairs": int(g.sum()), "PC": float(found[m].sum() / max(g.sum(), 1)),
                "entity_completeness": float((found[m][g > 0] >= g[g > 0]).mean()) if (g > 0).any() else float("nan"),
                "RR": float(1 - nc[m].sum() / max(m.sum() * n_pool, 1)), "PQ": float(found[m].sum() / max(nc[m].sum(), 1)),
                "cand_per_s1_mean": float(nc[m].mean()), "cand_per_s1_p95": float(np.percentile(nc[m], 95)),
                "cand_per_s1_max": int(nc[m].max())}
    out = one(np.ones(n_s1, bool))
    if country is not None:
        out["by_country"] = {c: one(country == c) for c in sorted(set(country))}
    return out


# ------------------------------------------------------------------ main
def fit_predict_scale(data_dir, cfg, report, out_dir=None, tracker=None, ckpt=None):
    T = tracker or Tracker(report)
    ck = ckpt or NULL
    sc = cfg.get("scale", {})
    bcfg = dict(cfg["blocking"])
    bcfg.update(sc.get("blocking", {}))
    seed = cfg["seed"]
    chunk_s1 = int(sc.get("chunk_s1", 100000))
    top_m = int(cfg.get("collective", {}).get("top_m", 10))
    cache_dir = Path(ck.root) if ck.enabled else Path(out_dir or ".") / "norm_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    d = Path(data_dir)

    with T.stage("normalize_sources"):
        paths = {split: {s: normalized_source(d / split / f"{split}_source{s}.tsv", cache_dir) for s in (1, 2, 3)}
                 for split in ("train", "test")}
    backend = detect_backend(cfg.get("backend", "auto"))
    seeds = cfg["seeds_gpu"] if backend == "xgboost_cuda" else cfg["seeds_cpu"]
    report["backend"], report["seeds"], report["mode"] = backend, list(seeds), "scale"
    parts_tr = partition_keys([paths["train"][1]])
    parts_te = partition_keys([paths["test"][1]])
    report["partitions"] = {"train": parts_tr, "test": parts_te}

    # ---------------- train sample (integer codes) + its gold
    with T.stage("train_sample"):
        s1_meta = pq.read_table(paths["train"][1], columns=["id", "cty"]).to_pandas()
        rng = np.random.RandomState(seed)
        n_take = min(int(sc.get("train_s1", 300000)), len(s1_meta))
        take = np.sort(rng.choice(len(s1_meta), n_take, replace=False))
        ids_s = np.array(sorted(s1_meta["id"].to_numpy(object)[take]), dtype=object)     # code -> S1 id
        code_of = pd.Index(ids_s)
        cty_s = pd.Series(s1_meta["cty"].to_numpy(object), index=s1_meta["id"].to_numpy(object)).reindex(ids_s)
        cty_s = cty_s.fillna("?").to_numpy(object)
        gold_all = load_gold(d / "train" / "train_ground_truth.tsv")
        gold = {s: gold_all.get(s, []) for s in ids_s}
        del gold_all, s1_meta
        G = np.array([len(gold[s]) for s in ids_s])
        strata = np.array([f"{c}_{min(g, 3)}" for c, g in zip(cty_s, G)], dtype=object)
        fold_s1 = make_folds(strata, cfg["n_folds"], seed)
        log.info("train sample: %d S1 entities, %d true pairs", len(ids_s), int(G.sum()))

    # ---------------- 1. train blocking (sample S1 vs full pool of the partition)
    k_union = ck.key("scale-union", {s: str(p) for s, p in paths["train"].items()}, bcfg, len(ids_s), seed,
                     code_hash("bigblock", "blocking"))                    # union depends on blocking code only
    UB = ck.load(k_union)
    if UB is None:
        Us, idfs, n_pool = [], {}, {}
        for part in parts_tr:
            with T.stage(f"train_blocking[{part}]"):
                s1 = load_partition([paths["train"][1]], part)
                pool = load_partition([paths["train"][2], paths["train"][3]], part)
                ids1 = s1["id"].to_numpy(object)
                idsp = pd.Index(pool["id"].to_numpy(object))
                idx = PartitionIndex(s1, pool, bcfg)
                idfs[part], n_pool[part] = idx.idf, len(pool)
                code_rows = code_of.get_indexer(ids1)
                rows = np.flatnonzero(code_rows >= 0)
                # gold pairs of this partition as int64 keys (s1 code * n_pool + pool position)
                gs, gc = [], []
                for r in rows:
                    for c in gold[ids1[r]]:
                        gs.append(code_rows[r])
                        gc.append(c)
                gpos = idsp.get_indexer(gc)
                gkey = np.unique(np.asarray(gs, np.int64)[gpos >= 0] * len(pool) + gpos[gpos >= 0])
                chs = _chunks(rows, chunk_s1)
                for ci, ch in enumerate(chs):
                    u = idx.query(ch)
                    if u is None:
                        continue
                    u["s1loc"] = u["s1"].astype(np.int32)
                    u["s1"] = code_rows[u["s1"].to_numpy()].astype(np.int32)
                    u["cand"] = u["cand"].astype(np.int32)
                    u["y"] = np.isin(u["s1"].to_numpy(np.int64) * len(pool) + u["cand"].to_numpy(np.int64), gkey)
                    u["part"] = part
                    Us.append(u)
                    log.info("    %s chunk %d/%d: %d S1 -> %d union pairs, %d true (RSS %s GB)", part, ci + 1, len(chs),
                             len(ch), len(u), int(u["y"].sum()), rss_gb())
                del idx, s1, pool, idsp
        UB = {"U": pd.concat(Us, ignore_index=True), "idfs": idfs, "n_pool": n_pool}
        del Us
        ck.save(k_union, UB)
    U, idfs, n_pool = UB["U"], UB["idfs"], UB["n_pool"]
    del UB                                     # the container would keep the ~4 GB union alive until the end
    U["part"] = U["part"].astype("category")

    # ---------------- 2. meta-blocker
    mc = cfg["meta"]
    k_meta = ck.key("scale-meta", k_union, mc, _small(cfg, backend), backend, seed, code_hash("models"))
    with T.stage("meta_blocker"):
        yU = U["y"].to_numpy()
        s1U = U["s1"].to_numpy()
        rate = float(mc.get("neg_sample", 0.25))
        mask = yU | (np.random.RandomState(seed).rand(len(yU)) < rate)
        meta_oof, _, minfo = fit_oof(U[B.META_FEATS].to_numpy(np.float32), yU, fold_s1[s1U], None, B.META_FEATS, cfg,
                                     backend, seeds[:1], name="meta_blocker", params=_small(cfg, backend), rounds=1500,
                                     es=50, ckpt=ck, ckpt_key=k_meta, train_mask=mask, keep_models=True)
        meta_models = minfo.pop("_models")
        minfo.pop("_importance_all", None)
        tau = B.choose_tau(s1U, meta_oof, yU, mc["top_keep"], mc["cap"], mc["recall_ratio"])
        keep = B.select_candidates(s1U, meta_oof, mc["top_keep"], tau, mc["cap"])
        U["meta"] = meta_oof.astype(np.float32)
        report["meta_blocker"] = {**minfo, "tau": tau}
        npool_tot = sum(n_pool.values())
        uniq = {}
        for ch in B.CHANNELS:
            hit = U[f"s_{ch}"].to_numpy() > 0
            only = hit & (U["n_ch"].to_numpy() == 1)
            uniq[ch] = {"true_pairs_found": int((yU & hit).sum()), "unique_true_pairs": int((yU & only).sum())}
        report["blocking"] = {"train_union": {**kpis(s1U, yU, len(ids_s), G, npool_tot, cty_s),
                                              "unique_true_pairs_by_channel": uniq},
                              "train_final": kpis(s1U[keep], yU[keep], len(ids_s), G, npool_tot, cty_s)}
        P_tr = U[keep].reset_index(drop=True)
        del U, meta_oof, mask
        log.info("train sample blocking: PC union %.4f -> final %.4f | cand/S1 %.1f (union %.1f) | tau %.4g",
                 report["blocking"]["train_union"]["PC"], report["blocking"]["train_final"]["PC"],
                 report["blocking"]["train_final"]["cand_per_s1_mean"],
                 report["blocking"]["train_union"]["cand_per_s1_mean"], tau)

    # ---------------- 3. train features + pass-1
    k_feat = ck.key("scale-feat", k_meta, float(tau), code_hash("features", "normalize", "bigblock"))
    with T.stage("train_features"):
        FX = ck.load(k_feat)
        if FX is None:
            Xs, Ps, Es, off = [], [], [], 0
            for part in parts_tr:
                s1 = load_partition([paths["train"][1]], part)
                pool = load_partition([paths["train"][2], paths["train"][3]], part)
                stats = partition_stats(s1, pool)
                sub = P_tr[P_tr["part"] == part].reset_index(drop=True)
                sub_in = sub[["s1loc", "cand"] + [c for c in sub.columns if c.startswith(("s_", "k_"))] +
                             ["n_ch", "meta"] + [c for _, c in B.EXACT]].rename(columns={"s1loc": "s1"})
                Xs.append(_features_batched(sub_in, s1, pool, idfs[part], stats, part))
                Ps.append(sub)
                nm, ad = _cand_strings(pool, sub["cand"].to_numpy())
                I, J, sim, sn = sibling_edges(sub["s1"].to_numpy(), sub["meta"].to_numpy(), nm, ad, top_m)
                Es.append((I + off, J + off, sim, sn))
                off += len(sub)
                log.info("    %s: features for %d train pairs (RSS %s GB)", part, len(sub), rss_gb())
                del s1, pool, stats, sub_in
            FX = (pd.concat(Xs, ignore_index=True), pd.concat(Ps, ignore_index=True),
                  tuple(np.concatenate([e[i] for e in Es]) for i in range(4)))
            ck.save(k_feat, FX)
        X_tr, P_tr, E_tr = FX
        del FX                                 # same: release the tuple so X_tr can actually be freed later
    if sc.get("train_only"):                   # training data ready; models are fit by src.bigtrain
        report["train_only"] = {"train_pairs": int(len(P_tr)), "features": list(X_tr.columns)}
        log.info("train_only: %d training pairs x %d features ready; stopping before the models", len(P_tr),
                 X_tr.shape[1])
        return None, None
    feats = list(X_tr.columns)
    y = P_tr["y"].to_numpy().astype(np.float32)
    s1_tr = P_tr["s1"].to_numpy(np.int64)
    fold_pair = fold_s1[s1_tr]
    mono = monotone_vector(feats) if cfg["model"].get("monotone", True) else None
    k_p1 = ck.key("scale-pass1", k_feat, cfg["model"], backend, list(seeds), code_hash("models"))
    with T.stage("pass1"):
        p1_oof, _, info1 = fit_oof(X_tr.to_numpy(np.float32), y, fold_pair, None, feats, cfg, backend, seeds, mono,
                                   name="pass1", ckpt=ck, ckpt_key=k_p1, keep_models=True)
        p1_models = info1.pop("_models")
        imp1 = info1.pop("_importance_all", {})
        report.setdefault("models", {})["pass1"] = info1

    # ---------------- 3b. pass-2: pass-1 features + within-S1 collective features (sibling support etc.)
    k_p2 = ck.key("scale-pass2", k_p1, top_m, code_hash("models", "collective"))
    with T.stage("pass2"):
        C_tr = within_collective(s1_tr, p1_oof, E_tr)
        f2 = feats + WITHIN
        mono2 = (mono + [0] * len(WITHIN)) if mono else None
        p2_oof, _, info2 = fit_oof(np.hstack([X_tr.to_numpy(np.float32), C_tr.to_numpy(np.float32)]), y, fold_pair,
                                   None, f2, cfg, backend, seeds, mono2, name="pass2", ckpt=ck, ckpt_key=k_p2,
                                   keep_models=True)
        p2_models = info2.pop("_models")
        imp2 = info2.pop("_importance_all", {})
        report["models"]["pass2"] = info2
        del C_tr

    # ---------------- 4. decoder (no entity model in the scale path)
    dc = dict(cfg["decoder"], use_h=False)
    with T.stage("decoder_tuning"):
        cand_codes = P_tr["cand"].to_numpy(np.int64) + P_tr["part"].cat.codes.to_numpy(np.int64) * (1 << 32)
        h0 = np.zeros(len(ids_s))
        hl = np.zeros(len(ids_s), np.float32)
        np.maximum.at(hl, s1_tr, y)
        rep1, fitted1 = tune_decoder(s1_tr, cand_codes, p1_oof, y, h0, hl, fold_s1, G, len(ids_s), dc)
        rep, fitted = tune_decoder(s1_tr, cand_codes, p2_oof, y, h0, hl, fold_s1, G, len(ids_s), dc)
        gain = rep["nested_f05"] - rep1["nested_f05"]
        use_p2 = gain >= float(sc.get("pass2_min_gain", 0.001))            # plan's kill rule
        report["variants_nested_f05"] = {"pass1": rep1["nested_f05"], "pass2": rep["nested_f05"]}
        report["chosen_variant"] = "pass2" if use_p2 else "pass1"
        log.info("nested OOF F0.5: pass-1 %.5f | pass-2 %.5f (gain %+.5f) -> using %s", rep1["nested_f05"],
                 rep["nested_f05"], gain, report["chosen_variant"])
        if not use_p2:
            rep, fitted, p2_oof = rep1, fitted1, p1_oof
        report["decoder"] = rep
        k_o, IDX_o, _ = decode(s1_tr, cand_codes, p2_oof, h0, len(ids_s), dc, fitted)
        # evaluate with string ids (only the sample's predicted/candidate ids are materialised)
        idmap = {}
        for part in parts_tr:
            arr = np.concatenate([pq.read_table(paths["train"][s], columns=["id"], filters=[("cty", "==", part)])
                                  .column(0).to_numpy(zero_copy_only=False) for s in (2, 3)])
            idmap[part] = arr
        cand_str = np.empty(len(P_tr), dtype=object)
        for part in parts_tr:
            m = (P_tr["part"] == part).to_numpy()
            cand_str[m] = idmap[part][P_tr["cand"].to_numpy()[m]]
        del idmap
        pred_oof = {ids_s[i]: cand_str[r].tolist() for i, r in enumerate(selections(k_o, IDX_o))}
        order = np.lexsort((-P_tr["meta"].to_numpy(), s1_tr))
        cf = {}
        so = s1_tr[order]
        bnd = np.flatnonzero(np.r_[True, so[1:] != so[:-1], True])
        for a, b in zip(bnd[:-1], bnd[1:]):
            cf[ids_s[so[a]]] = cand_str[order[a:b]].tolist()
        country = dict(zip(ids_s, cty_s))
        report["oof"] = evaluate(pred_oof, gold, country, cf)
        report["oof"]["nested_f05"] = rep["nested_f05"]
        report["calibration"] = {"pass1_oof": {**binary_metrics(y, p1_oof), **calibration_report(y, p1_oof)},
                                 "pass2_oof_raw": {**binary_metrics(y, p2_oof), **calibration_report(y, p2_oof)},
                                 "pass2_oof_isotonic": calibration_report(y, fitted["iso"].predict(p2_oof))}
        log.info("OOF (train sample) macro F0.5: nested %.5f | final policy %.5f | by country %s",
                 rep["nested_f05"], report["oof"]["macro_f05"],
                 {k: round(v["f05"], 4) for k, v in report["oof"].get("by_country", {}).items()})
        log.info("OOF error budget: %s", report["oof"].get("error_budget"))
    if out_dir is not None:
        _artifacts(Path(out_dir), X_tr, P_tr, ids_s, cand_str, p2_oof, fitted, {**imp1, **{f"pass2:{k}": v for k, v in imp2.items()}})
    # everything training-side is done: free it before the largest test partition is indexed (~7 GB for India)
    del X_tr, cand_str, P_tr, E_tr, pred_oof, cf, p1_oof, p2_oof, y, s1_tr, fold_pair, cand_codes, h0, hl
    import gc
    gc.collect()
    log.info("training objects released before test (RSS %s GB)", rss_gb())

    # ---------------- 5. test: chunked blocking -> meta -> features -> p1
    per_part = []
    for part in parts_te:
        with T.stage(f"test[{part}]"):
            s1 = load_partition([paths["test"][1]], part)
            pool = load_partition([paths["test"][2], paths["test"][3]], part)
            ids1, idsp = s1["id"].to_numpy(object), pool["id"].to_numpy(object)
            # test predictions depend on the blocking/feature/collective code and the fitted models, not on this
            # orchestration file (so memory/logging fixes here do not throw away finished test chunks)
            k_part = ck.key("scale-test", part, str(paths["test"][1]), str(paths["test"][2]), str(paths["test"][3]),
                            k_p1, k_p2, bcfg, float(tau), mc, code_hash("bigblock", "features", "collective"), "v2")
            res = ck.load(k_part)
            if res is None:
                idx = PartitionIndex(s1, pool, bcfg)
                stats = partition_stats(s1, pool)
                chs = _chunks(np.arange(len(s1)), chunk_s1)
                acc, t0 = [], time.time()
                for ci, ch in enumerate(chs):
                    kc = f"{k_part}-chunk{ci}"
                    r = ck.load(kc)
                    if r is None:
                        r = score_chunk(idx, ch, s1, pool, stats, part, (meta_models, p1_models, p2_models), feats,
                                        tau, mc, top_m)
                        if r is None:
                            continue
                        ck.save(kc, r)
                    acc.append(r)
                    el = time.time() - t0
                    log.info("    %s chunk %d/%d: %d candidates (%.1f/S1), mean p1 %.3f | %.0fs, ETA %.0fs, RSS %s GB",
                             part, ci + 1, len(chs), len(r[0]), len(r[0]) / max(len(ch), 1),
                             float(r[2].mean()) if len(r[2]) else 0.0, el, el / (ci + 1) * (len(chs) - ci - 1),
                             rss_gb())
                res = tuple(np.concatenate([a[i] for a in acc]) if acc else np.zeros(0) for i in range(5))
                ck.save(k_part, res)
                del idx, stats
            per_part.append((part, ids1, idsp, res))
            del s1, pool

    # ---------------- 6. exclusivity + decoding over all test pairs
    with T.stage("decode_test"):
        ids1 = np.concatenate([x[1] for x in per_part])
        idsp = np.concatenate([x[2] for x in per_part])
        off1 = np.cumsum([0] + [len(x[1]) for x in per_part])
        offp = np.cumsum([0] + [len(x[2]) for x in per_part])
        s_all = np.concatenate([x[3][0].astype(np.int64) + off1[i] for i, x in enumerate(per_part)])
        c_all = np.concatenate([x[3][1].astype(np.int64) + offp[i] for i, x in enumerate(per_part)])
        p_all = np.concatenate([x[3][2] if use_p2 else x[3][4] for x in per_part])
        m_all = np.concatenate([x[3][3] for x in per_part])
        k_t, IDX_t, P_t = decode(s_all, c_all, p_all, np.zeros(len(ids1)), len(ids1), dc, fitted)
        pred = {ids1[i]: idsp[c_all[r]].tolist() for i, r in enumerate(selections(k_t, IDX_t))}
        order = np.lexsort((-m_all, s_all))
        ss, cc = s_all[order], c_all[order]
        bnd = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1], True])
        cand = {ids1[ss[a]]: idsp[cc[a:b]].tolist() for a, b in zip(bnd[:-1], bnd[1:])}
        dash = {}
        for i, (part, pid1, pidp, res) in enumerate(per_part):
            mk = np.zeros(len(ids1), bool)
            mk[off1[i]:off1[i + 1]] = True
            has = np.zeros(len(pid1), bool)
            has[np.asarray(res[0], np.int64)] = True
            dash[part] = {"n_s1": int(len(pid1)), "n_pool": int(len(pidp)), "candidates": int(len(res[0])),
                          "no_candidates_rate": float(1 - has.mean()) if len(pid1) else 0.0,
                          "cand_per_s1": len(res[0]) / max(len(pid1), 1),
                          "pred_empty_rate": float((k_t[mk] == 0).mean()), "mean_matches": float(k_t[mk].mean()),
                          "top1_prob_hist": np.histogram(P_t[mk, 0], bins=10, range=(0, 1))[0].tolist()}
            log.info("dashboard %-8s S1=%8d cand/S1 %.1f pred-empty %.3f mean matches %.3f", part, dash[part]["n_s1"],
                     dash[part]["cand_per_s1"], dash[part]["pred_empty_rate"], dash[part]["mean_matches"])
        report["test_dashboard"] = {"test": dash}
        report["blocking"]["test_final"] = {"pairs": int(len(s_all)), "cand_per_s1_mean": len(s_all) / max(len(ids1), 1),
                                            "RR": 1 - len(s_all) / max(len(ids1) * len(idsp), 1)}
    return pred, cand


def score_chunk(idx, rows, s1, pool, stats, part, models, feats, tau, mc, top_m, union=False):
    """The test path for one chunk of S1 rows (positions in idx's S1 frame): blocking union -> meta-blocker top-K
    -> pair features -> pass-1 -> within-S1 collective -> pass-2, every model being the mean of its fold models.
    Returns (s1, cand, p2, meta, p1) or None without candidates; union=True also returns the whole union as
    (s1, cand, meta) so that blocking recall beyond the cap can be measured (proxy only)."""
    meta_models, p1_models, p2_models = models
    u = idx.query(rows)
    if u is None:
        return None
    m = predict_models(meta_models, u[B.META_FEATS].to_numpy(np.float32))
    kp = B.select_candidates(u["s1"].to_numpy(), m, mc["top_keep"], tau, mc["cap"])
    Pc = u[kp].reset_index(drop=True)
    Pc["meta"] = m[kp].astype(np.float32)
    X = _features_batched(Pc, s1, pool, idx.idf, stats, part)
    Xn = X[feats].to_numpy(np.float32)
    del X
    p1 = predict_models(p1_models, Xn)
    nm, ad = _cand_strings(pool, Pc["cand"].to_numpy())
    C = within_collective(Pc["s1"].to_numpy(), p1,
                          sibling_edges(Pc["s1"].to_numpy(), Pc["meta"].to_numpy(), nm, ad, top_m))
    p = predict_models(p2_models, np.hstack([Xn, C.to_numpy(np.float32)]))
    r = (Pc["s1"].to_numpy(np.int64), Pc["cand"].to_numpy(np.int64), p.astype(np.float32),
         Pc["meta"].to_numpy(np.float32), p1.astype(np.float32))
    if union:
        r = r + ((u["s1"].to_numpy(np.int64), u["cand"].to_numpy(np.int64), m.astype(np.float32)),)
    return r


def load_fold_models(ck, key, n_folds):
    """Fold models kept by fit_oof(keep_models=True), read straight from their per-fold checkpoints."""
    models = []
    for f in range(n_folds):
        saved = ck.load(f"{key}-fold{f}")
        if saved is None or not saved.get("models"):
            raise FileNotFoundError(f"no fold models in checkpoint {key}-fold{f}")
        models.append(("avg", saved["models"], None))
    return models


def _artifacts(out, X_tr, P_tr, ids_s, cand_str, p1_oof, fitted, imp1):
    try:
        from .pipeline import KEY_FEATS
        out.mkdir(parents=True, exist_ok=True)
        d = pd.DataFrame({"s1_id": ids_s[P_tr["s1"].to_numpy()], "cand_id": cand_str,
                          "country": P_tr["part"].astype(str).to_numpy(), "y": P_tr["y"].to_numpy().astype(np.int8),
                          "p1": p1_oof, "p1_cal": fitted["iso"].predict(p1_oof)})
        for c in KEY_FEATS:
            if c in X_tr:
                d[c] = X_tr[c].to_numpy()
        write_table(d, out / "oof_pairs")
        pd.Series(imp1, name="pass1_gain").rename_axis("feature").reset_index().to_csv(out / "feature_importance.csv",
                                                                                       index=False)
        pd.DataFrame([{"oof_f05_in_sample": f, "config": str(c)} for f, c in fitted.get("grid", [])]).to_csv(
            out / "decoder_grid.csv", index=False)
    except Exception as e:  # diagnostics must never break a run
        log.warning("could not write artifacts: %r", e)


def test_ids(data_dir):
    """Test S1 ids in file order (the writer's row order) and the set of valid pool ids, read id-column only."""
    d = Path(data_dir) / "test"
    s1 = _read_arrow(d / "test_source1.tsv").column("entity_id").to_pylist()
    pool = []
    for k in (2, 3):
        pool += _read_arrow(d / f"test_source{k}.tsv").column("entity_id").to_pylist()
    return [s.strip() for s in s1 if s.strip()], pool
