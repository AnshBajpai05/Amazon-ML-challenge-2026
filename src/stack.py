"""Second-stage stacker: re-scores every candidate pair with its group and competition context (+ cross-encoder).

For each pair the pass-2 LightGBM gives p2 in isolation (plus within-S1 summaries). The stacker sees, per pair:
  * the model scores: logit p2, logit p1, meta-blocker score;
  * group context of its S1 (what set attention over the S1's candidates would use): rank of the pair in the S1,
    gap to the S1's best, the S1's probability mass, number of strong / medium candidates, candidate count;
  * competition for its pool record: strongest rival S1 claim, number of rival claims > 0.5, rank of this claim;
  * optional cross-encoder logits (Kaggle kit, one column per model) with a has-score flag.
It is trained on the test-like holdout (src/proxy.py) with city-grouped folds, so its out-of-fold holdout
probabilities are honest inputs for the decoder tuning, and the model fitted on the whole holdout scores test.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .blocking import group_rank
from .compat import effective_cpus

log = logging.getLogger("ber")
STACK_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                    feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, verbose=-1, seed=11)
STACK_ROUNDS = 600


def _logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def stack_features(s, c, p2, p1, meta, ce=None):
    """Per-pair feature matrix (float32) and names. ce: dict name -> array of logits with NaN where unscored."""
    s, c = np.asarray(s), np.asarray(c)
    p = np.asarray(p2, np.float64)
    n = int(s.max()) + 1 if len(s) else 0
    rk = group_rank(s, p, tie=c)
    top = np.zeros(n)
    np.maximum.at(top, s, p)
    mass = np.bincount(s, weights=p, minlength=n)
    strong = np.bincount(s, weights=(p > 0.5).astype(np.float64), minlength=n)
    mid = np.bincount(s, weights=((p > 0.2) & (p <= 0.8)).astype(np.float64), minlength=n)
    cnt = np.bincount(s, minlength=n).astype(np.float64)
    _, inv = np.unique(c, return_inverse=True)
    rank_c = group_rank(inv, p, tie=s)
    order = np.lexsort((-p, inv))
    ii = inv[order]
    first = np.r_[True, ii[1:] != ii[:-1]]
    second = np.r_[False, first[:-1] & ~first[1:]]
    m = inv.max() + 1 if len(inv) else 0
    top_c, sec_c = np.zeros(m), np.zeros(m)
    top_c[ii[first]] = p[order][first]
    sec_c[ii[second]] = p[order][second]
    rival = np.where(rank_c == 1, sec_c[inv], top_c[inv])
    n_riv = np.bincount(inv, weights=(p > 0.5).astype(np.float64), minlength=m)[inv] - (p > 0.5)
    cols = {"lp2": _logit(p), "lp1": _logit(p1), "meta": np.asarray(meta, np.float64), "rank_s1": rk,
            "gap_top": top[s] - p, "rel_top": p / np.maximum(top[s], 1e-9), "s1_mass": mass[s],
            "s1_strong": strong[s], "s1_mid": mid[s], "s1_cnt": cnt[s], "rival_p": rival, "rival_n": n_riv,
            "claim_rank": rank_c, "margin_rival": p - rival}
    for k, v in (ce or {}).items():
        v = np.asarray(v, np.float64)
        cols[k] = v
        cols[f"has_{k}"] = (~np.isnan(v)).astype(np.float64)
        cols[f"{k}_x_lp2"] = np.where(np.isnan(v), np.nan, v - cols["lp2"])
    names = list(cols)
    return np.column_stack([cols[k] for k in names]).astype(np.float32), names


def fit_stacker(X, y, folds, names):
    """OOF probabilities over the given folds + a model fitted on all rows."""
    import lightgbm as lgb
    prm = dict(STACK_PARAMS, num_threads=effective_cpus())
    oof = np.zeros(len(y))
    for f in np.unique(folds):
        tr = folds != f
        b = lgb.train(prm, lgb.Dataset(X[tr], y[tr], feature_name=names), STACK_ROUNDS)
        oof[~tr] = b.predict(X[~tr])
    full = lgb.train(prm, lgb.Dataset(X, y, feature_name=names), STACK_ROUNDS)
    imp = dict(zip(names, full.feature_importance("gain")))
    tot = sum(imp.values()) or 1.0
    log.info("stacker importance: %s", {k: round(v / tot, 3) for k, v in sorted(imp.items(), key=lambda kv: -kv[1])[:12]})
    return oof, full


def load_ce(path):
    """Cross-encoder scores from ce_outputs.zip (or its extracted folder): {split: DataFrame(s1_id, cand_id, ce_*)}."""
    import zipfile
    path = Path(path)
    out = {}
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                for split in ("holdout", "test"):
                    if name.endswith(f"scores_{split}.parquet"):
                        with z.open(name) as fh:
                            d = pd.read_parquet(fh)
                        out.setdefault(split, []).append(d)
    else:
        for split in ("holdout", "test"):
            for f in sorted(path.rglob(f"scores_{split}.parquet")):
                out.setdefault(split, []).append(pd.read_parquet(f))
    res = {}
    for split, ds in out.items():
        m = None
        for d in ds:
            keep = ["s1_id", "cand_id"] + [c for c in d.columns if c.startswith("ce_")]
            d = d[keep].drop_duplicates(["s1_id", "cand_id"])
            m = d if m is None else m.merge(d, on=["s1_id", "cand_id"], how="outer")
        res[split] = m
        log.info("cross-encoder %s scores: %d pairs, models %s", split, len(m), [c for c in m.columns if c.startswith("ce_")])
    return res


def attach_ce(s1_ids, cand_ids, ce_df):
    """Align cross-encoder logits to the given pairs: dict ce_name -> array (NaN where not scored)."""
    if ce_df is None or not len(ce_df):
        return None
    key = pd.MultiIndex.from_arrays([pd.Index(s1_ids, dtype=object), pd.Index(cand_ids, dtype=object)])
    ce = ce_df.set_index(["s1_id", "cand_id"])
    ce = ce[~ce.index.duplicated()]
    pos = ce.index.get_indexer(key)
    out = {}
    for col in ce.columns:
        v = ce[col].to_numpy(np.float64)
        out[col] = np.where(pos >= 0, v[np.maximum(pos, 0)], np.nan)
        log.info("  %s: %.1f%% of pairs scored", col, 100 * (pos >= 0).mean())
    return out
