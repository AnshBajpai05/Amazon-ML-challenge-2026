"""Pass-2 collective features computed from pass-1 probabilities (plan 7.3, A.6).

Train uses OUT-OF-FOLD p1, test uses the fold-mean p1, so both sides see the same kind of input.
All operations are vectorised group-bys (no per-entity Python loops).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .blocking import group_rank
from .compat import cpdist

COLLECTIVE = ["p1", "rank_in_s1", "gap_to_s1_best", "p1_rel", "s1_p1_max", "s1_p1_2nd", "s1_p1_sum",
              "n_strong_in_s1", "n_mid_in_s1", "rank_in_cand", "n_s1_for_cand", "margin_vs_other_s1",
              "is_best_s1_for_cand", "cand_p1_sum", "recip_best", "sib_support", "sib_count", "sib_name_sim"]


def _second(keys, vals, rank):
    """Second-best value per group (0 when the group has one member)."""
    v2 = np.where(rank == 2, vals, -1.0)
    s = pd.Series(v2).groupby(keys).transform("max").to_numpy()
    return np.where(s < 0, 0.0, s)


def add_collective(pairs: pd.DataFrame, p1: np.ndarray, npool, top_m=10) -> pd.DataFrame:
    s1 = pairs["s1"].to_numpy()
    cand = pairs["cand"].to_numpy()
    p = np.asarray(p1, np.float64)
    out = {"p1": p}
    r1 = group_rank(s1, p, tie=cand)
    g1 = pd.Series(p).groupby(s1)
    mx1 = g1.transform("max").to_numpy()
    out["rank_in_s1"] = r1
    out["gap_to_s1_best"] = mx1 - p
    out["p1_rel"] = p / np.maximum(mx1, 1e-9)
    out["s1_p1_max"] = mx1
    out["s1_p1_2nd"] = _second(s1, p, r1)
    out["s1_p1_sum"] = g1.transform("sum").to_numpy()
    out["n_strong_in_s1"] = pd.Series(p > 0.5).groupby(s1).transform("sum").to_numpy()
    out["n_mid_in_s1"] = pd.Series(p > 0.2).groupby(s1).transform("sum").to_numpy()
    rc = group_rank(cand, p, tie=s1)
    gc = pd.Series(p).groupby(cand)
    best = gc.transform("max").to_numpy()
    second = _second(cand, p, rc)
    out["rank_in_cand"] = rc
    out["n_s1_for_cand"] = gc.transform("count").to_numpy()
    out["margin_vs_other_s1"] = np.where(rc == 1, p - second, p - best)
    out["is_best_s1_for_cand"] = (rc == 1).astype(np.float32)
    out["cand_p1_sum"] = gc.transform("sum").to_numpy()
    out["recip_best"] = ((rc == 1) & (r1 == 1)).astype(np.float32)
    sup, cnt, nsim = sibling_support(s1, cand, p, r1, npool, top_m)
    out["sib_support"], out["sib_count"], out["sib_name_sim"] = sup, cnt, nsim
    return pd.DataFrame({k: np.asarray(v, np.float32) for k, v in out.items()})


def sibling_support(s1, cand, p, r1, npool, top_m=10):
    """max over the S1's other top candidates c' of p1(s1,c') * sim(c, c').
    sim = mean of token-set ratios on name and address (rapidfuzz, C++)."""
    n = len(s1)
    sup = np.zeros(n, np.float32)
    cnt = np.zeros(n, np.float32)
    nsim = np.zeros(n, np.float32)
    sel = np.flatnonzero(r1 <= top_m)
    if len(sel) < 2:
        return sup, cnt, nsim
    # group the selected rows by S1 in rank order
    order = sel[np.lexsort((r1[sel], s1[sel]))]
    ks = s1[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    sizes = np.diff(np.r_[starts, len(order)])
    I, J = [], []
    for g in range(2, int(sizes.max()) + 1 if len(sizes) else 0):
        st = starts[sizes == g]
        if not len(st):
            continue
        a, b = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
        m = a != b
        I.append((st[:, None] + a[m][None, :]).ravel())
        J.append((st[:, None] + b[m][None, :]).ravel())
    if not I:
        return sup, cnt, nsim
    I, J = order[np.concatenate(I)], order[np.concatenate(J)]
    name = npool["n_core"].where(npool["n_core"].str.len() > 0, npool["n_full"]).to_numpy(object)
    addr = npool["a_full"].to_numpy(object)
    ci, cj = cand[I], cand[J]
    sn = cpdist(name[ci].tolist(), name[cj].tolist(), fuzz.token_set_ratio) / 100.0
    sa = cpdist(addr[ci].tolist(), addr[cj].tolist(), fuzz.token_set_ratio) / 100.0
    sim = 0.5 * (sn + sa)
    np.maximum.at(sup, I, (p[J] * sim).astype(np.float32))
    np.add.at(cnt, I, ((p[J] > 0.5) & (sim > 0.6)).astype(np.float32))
    np.maximum.at(nsim, I, sn.astype(np.float32))
    return sup, cnt, nsim


ENTITY_FEATS = ["e_top1", "e_top2", "e_gap", "e_sum", "e_entropy", "e_n03", "e_n05", "e_n07", "e_ncand",
                "e_max_ncos", "e_max_acos", "e_max_jcos", "e_top1_recip", "e_top1_nsd", "e_top1_asd",
                "e_top1_post", "e_top1_margin", "e_s1_ntok_n", "e_s1_ntok_a", "e_s1_has_post", "e_s1_name_freq",
                "e_s1_name_idf"]


def entity_features(pairs, p2, X, C, n1, n_s1, idf):
    """Per-S1 aggregates for the entity model h = P(at least one candidate is a true match)."""
    s1 = pairs["s1"].to_numpy()
    p = np.asarray(p2, np.float64)
    E = np.zeros((n_s1, len(ENTITY_FEATS)), np.float32)
    idx = {f: i for i, f in enumerate(ENTITY_FEATS)}
    if len(s1):
        r = group_rank(s1, p, tie=pairs["cand"].to_numpy())
        top = np.flatnonzero(r == 1)
        sec = np.flatnonzero(r == 2)
        us = s1[top]
        E[us, idx["e_top1"]] = p[top]
        E[s1[sec], idx["e_top2"]] = p[sec]
        E[:, idx["e_gap"]] = E[:, idx["e_top1"]] - E[:, idx["e_top2"]]
        g = pd.DataFrame({"s1": s1, "p": p, "ent": -p * np.log(np.clip(p, 1e-9, 1)),
                          "n03": p > 0.3, "n05": p > 0.5, "n07": p > 0.7,
                          "nc": X["n_cos_c"].to_numpy(), "ac": X["a_cos_c"].to_numpy(),
                          "jc": X["j_cos_c"].to_numpy()}).groupby("s1")
        agg = g.agg(sum=("p", "sum"), ent=("ent", "sum"), n03=("n03", "sum"), n05=("n05", "sum"),
                    n07=("n07", "sum"), n=("p", "size"), nc=("nc", "max"), ac=("ac", "max"), jc=("jc", "max"))
        ii = agg.index.to_numpy()
        for col, f in (("sum", "e_sum"), ("ent", "e_entropy"), ("n03", "e_n03"), ("n05", "e_n05"),
                       ("n07", "e_n07"), ("n", "e_ncand"), ("nc", "e_max_ncos"), ("ac", "e_max_acos"),
                       ("jc", "e_max_jcos")):
            E[ii, idx[f]] = agg[col].to_numpy(np.float32)
        E[us, idx["e_top1_recip"]] = C["recip_best"].to_numpy()[top]
        E[us, idx["e_top1_nsd"]] = X["n_sd"].to_numpy()[top]
        E[us, idx["e_top1_asd"]] = np.nan_to_num(X["a_sd"].to_numpy()[top])
        E[us, idx["e_top1_post"]] = X["post_eq2"].to_numpy()[top] - X["post_conf2"].to_numpy()[top]
        E[us, idx["e_top1_margin"]] = C["margin_vs_other_s1"].to_numpy()[top]
    E[:, idx["e_s1_ntok_n"]] = n1["n_core"].str.split().str.len().fillna(0).to_numpy(np.float32)
    E[:, idx["e_s1_ntok_a"]] = n1["a_core"].str.split().str.len().fillna(0).to_numpy(np.float32)
    E[:, idx["e_s1_has_post"]] = (n1["a_postal"].str.len() > 0).to_numpy(np.float32)
    key1 = (n1["cty"] + "\x01" + n1["n_core"])
    E[:, idx["e_s1_name_freq"]] = np.log1p(key1.map(key1.value_counts()).to_numpy(np.float32))
    mean_idf = []
    for cty, name in zip(n1["cty"], n1["n_core"]):
        nidf, _, nd, _ = idf.get(cty, idf.get("__all__", ({}, {}, 8.0, 8.0)))
        t = name.split()
        mean_idf.append(np.mean([nidf.get(x, nd) for x in t]) if t else 0.0)
    E[:, idx["e_s1_name_idf"]] = np.asarray(mean_idf, np.float32)
    return pd.DataFrame(E, columns=ENTITY_FEATS)


# ---------------------------------------------------------------- scale path: within-S1 collective features
# The large-data path trains on a sample of S1 entities, so competition *between* S1s (rank among the S1s
# claiming a pool record) cannot be reproduced on train; exclusivity is left to the decoder there. Everything
# below is complete within one S1's candidate set, hence identical in meaning for the train sample and test.
WITHIN = ["p1", "rank_in_s1", "gap_to_s1_best", "p1_rel", "s1_p1_max", "s1_p1_2nd", "s1_p1_sum",
          "n_strong_in_s1", "n_mid_in_s1", "sib_support", "sib_count", "sib_name_sim"]


def sibling_edges(s1, rank_key, names, addrs, top_m=10):
    """Ordered pairs (i, j) of rows of the same S1 among its top_m candidates by rank_key, with the name and
    joint (name+address) similarity of the two candidate records. names/addrs: per-row candidate strings."""
    s1 = np.asarray(s1)
    r = group_rank(s1, np.asarray(rank_key, np.float64))
    sel = np.flatnonzero(r <= top_m)
    z = (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32), np.zeros(0, np.float32))
    if len(sel) < 2:
        return z
    order = sel[np.lexsort((r[sel], s1[sel]))]
    ks = s1[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    sizes = np.diff(np.r_[starts, len(order)])
    I, J = [], []
    for g in range(2, int(sizes.max()) + 1):
        st = starts[sizes == g]
        if not len(st):
            continue
        a, b = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
        m = a != b
        I.append((st[:, None] + a[m][None, :]).ravel())
        J.append((st[:, None] + b[m][None, :]).ravel())
    if not I:
        return z
    I, J = order[np.concatenate(I)], order[np.concatenate(J)]
    names, addrs = np.asarray(names, dtype=object), np.asarray(addrs, dtype=object)
    sn = cpdist(names[I].tolist(), names[J].tolist(), fuzz.token_set_ratio) / 100.0
    sa = cpdist(addrs[I].tolist(), addrs[J].tolist(), fuzz.token_set_ratio) / 100.0
    return I.astype(np.int64), J.astype(np.int64), (0.5 * (sn + sa)).astype(np.float32), sn.astype(np.float32)


def within_collective(s1, p, edges):
    """Within-S1 collective features from probabilities p (OOF on train, fold-mean on test) + sibling edges."""
    s1 = np.asarray(s1)
    p = np.asarray(p, np.float64)
    n = len(p)
    r1 = group_rank(s1, p)
    g = pd.Series(p).groupby(s1)
    mx = g.transform("max").to_numpy()
    out = {"p1": p, "rank_in_s1": r1, "gap_to_s1_best": mx - p, "p1_rel": p / np.maximum(mx, 1e-9),
           "s1_p1_max": mx, "s1_p1_2nd": _second(s1, p, r1), "s1_p1_sum": g.transform("sum").to_numpy(),
           "n_strong_in_s1": pd.Series(p > 0.5).groupby(s1).transform("sum").to_numpy(),
           "n_mid_in_s1": pd.Series(p > 0.2).groupby(s1).transform("sum").to_numpy()}
    sup, cnt, nsim = np.zeros(n, np.float32), np.zeros(n, np.float32), np.zeros(n, np.float32)
    I, J, sim, sn = edges
    if len(I):
        np.maximum.at(sup, I, (p[J] * sim).astype(np.float32))
        np.add.at(cnt, I, ((p[J] > 0.5) & (sim > 0.6)).astype(np.float32))
        np.maximum.at(nsim, I, sn)
    out["sib_support"], out["sib_count"], out["sib_name_sim"] = sup, cnt, nsim
    return pd.DataFrame({k: np.asarray(v, np.float32) for k, v in out.items()})[WITHIN]
