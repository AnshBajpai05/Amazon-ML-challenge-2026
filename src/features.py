"""Stage 2: pair features (plan section 6), ~110 per (S1, candidate) pair.

Families: name strings, IDF soft alignment (abbreviation/typo aware), name semantics
(legal form, acronym, DBA variants, skeleton, digits), address strings + alignment,
numbers/postal, cross-field, rarity/chains, provenance (channel scores/ranks, meta score),
record flags. Never uses entity_id numbers, row order or a country one-hot.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, LCSseq, Levenshtein, Prefix

from .blocking import CHANNELS, EXACT
from .compat import cpdist, effective_cpus
from .normalize import acronym_match, soft_align, tok_sim

log = logging.getLogger("ber")

RF_SPECS = [
    ("n_jw", "nc", JaroWinkler.normalized_similarity, 1.0),
    ("n_lev", "nc", Levenshtein.normalized_similarity, 1.0),
    ("n_ratio", "nc", fuzz.ratio, 0.01),
    ("n_tsort", "nc", fuzz.token_sort_ratio, 0.01),
    ("n_tset", "nc", fuzz.token_set_ratio, 0.01),
    ("n_partial", "nc", fuzz.partial_ratio, 0.01),
    ("n_lcs", "nc", LCSseq.normalized_similarity, 1.0),
    ("n_prefix", "nc", Prefix.normalized_similarity, 1.0),
    ("nf_tset", "nf", fuzz.token_set_ratio, 0.01),
    ("nf_jw", "nf", JaroWinkler.normalized_similarity, 1.0),
    ("nr_tset", "nr", fuzz.token_set_ratio, 0.01),
    ("a_jw", "ac", JaroWinkler.normalized_similarity, 1.0),
    ("a_lev", "ac", Levenshtein.normalized_similarity, 1.0),
    ("a_tsort", "ac", fuzz.token_sort_ratio, 0.01),
    ("a_tset", "ac", fuzz.token_set_ratio, 0.01),
    ("a_partial", "ac", fuzz.partial_ratio, 0.01),
    ("a_lcs", "ac", LCSseq.normalized_similarity, 1.0),
    ("af_tset", "af", fuzz.token_set_ratio, 0.01),
    ("af_partial", "af", fuzz.partial_ratio, 0.01),
    ("j_tset", "jt", fuzz.token_set_ratio, 0.01),
    ("j_tsort", "jt", fuzz.token_sort_ratio, 0.01),
]

PY_FEATS = [
    "n_sd", "n_cov_a", "n_cov_b", "n_mu_a", "n_mu_b", "n_sh_max", "n_sh_min", "n_idf_a", "n_idf_b",
    "n_var_sd", "n_var_gain", "legal_eq", "legal_conf", "legal_miss1", "acronym", "skel_jacc",
    "nnum_eq", "nnum_conf", "n_first_eq", "n_first_sim", "n_last_sim",
    "a_sd", "a_cov_a", "a_cov_b", "a_mu_a", "a_mu_b", "a_sh_max", "af_sd",
    "a_ovl", "a_cont_a", "a_cont_b", "lm_cross",
    "num_jacc", "num_first_eq", "num_conf", "num_shared", "post_eq2", "post_conf2", "post_miss1",
    "name_in_addr_a", "name_in_addr_b",
    "ntok_n_a", "ntok_n_b", "ntok_a_a", "ntok_a_b", "a_empty_a", "a_empty_b",
]
NAN = float("nan")


def _sets(s, sep=" "):
    return set(x for x in s.split(sep) if x) if s else set()


# First-letter phonetic classes: tok_sim(a, b) > 0 requires a == b, an abbreviation or initial (same first
# letter), equal skeletons (skeleton keeps a phonetically mapped first letter: c/k/q/s/x/z, v/w/b, p/f, g/j)
# or Jaro-Winkler >= 0.9 (different first letters only reach that for long tokens). Everything else is 0,
# so it is rejected here without a call: ~10x fewer comparisons on long Indian addresses.
_FCLS = {c: g[0] for g in ("kcqsxz", "vwb", "pf", "jg") for c in g}
_MAX_ALIGN = 14


def fast_align(A, B, idf, default_idf):
    """soft_align with a first-letter-class fast reject and at most 14 tokens a side (same 9-tuple)."""
    A, B = A[:_MAX_ALIGN], B[:_MAX_ALIGN]
    if not A or not B:
        return soft_align(A, B, idf, default_idf)
    wA = [idf.get(t, default_idf) for t in A]
    wB = [idf.get(t, default_idf) for t in B]
    ca = [_FCLS.get(t[0], t[0]) for t in A]
    cb = [_FCLS.get(t[0], t[0]) for t in B]
    la = [len(t) for t in A]
    lb = [len(t) for t in B]
    cand = []
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            if a == b:
                s = 1.0
            elif (ca[i] != cb[j] and (la[i] < 8 or lb[j] < 8)) or a.isdigit() or b.isdigit():
                continue
            else:
                s = tok_sim(a, b)
                if s <= 0:
                    continue
            cand.append((s * (wA[i] + wB[j]) / 2, s, i, j))
    cand.sort(reverse=True)
    ua, ub, num_a, num_b, shared = set(), set(), 0.0, 0.0, []
    for _, s, i, j in cand:
        if i in ua or j in ub:
            continue
        ua.add(i)
        ub.add(j)
        num_a += s * wA[i]
        num_b += s * wB[j]
        if s >= 0.85:
            shared.append(min(wA[i], wB[j]))
    WA, WB = sum(wA) or 1e-9, sum(wB) or 1e-9
    return ((num_a + num_b) / (WA + WB), num_a / WA, num_b / WB,
            max([wA[i] for i in range(len(A)) if i not in ua], default=0.0),
            max([wB[j] for j in range(len(B)) if j not in ub], default=0.0),
            max(shared, default=0.0), min(shared, default=0.0), WA, WB)


def _py_chunk(rows, idf_maps):
    out = np.full((len(rows), len(PY_FEATS)), np.nan, np.float32)
    for i, (part, na, nb, va, vb, la, lb, ska, skb, nna, nnb, aa, ab, fa, fb, lma, lmb, ua, ub, pa, pb) in \
            enumerate(rows):
        nidf, aidf, nd, ad = idf_maps[part]
        A, B = na.split(), nb.split()
        sd = fast_align(A, B, nidf, nd)
        VA = [v.split() for v in va.split("|")] if va else [A]
        VB = [v.split() for v in vb.split("|")] if vb else [B]
        var_sd = sd[0]
        if va or vb:
            for x in VA:
                for y in VB:
                    var_sd = max(var_sd, fast_align(x, y, nidf, nd)[0])
        LA, LB = _sets(la, "|"), _sets(lb, "|")
        SA, SB = _sets(ska), _sets(skb)
        NA_, NB_ = _sets(nna), _sets(nnb)
        AA, AB = aa.split(), ab.split()
        FA, FB = fa.split(), fb.split()
        asd = fast_align(AA, AB, aidf, ad)
        afsd = fast_align(FA, FB, aidf, ad)[0] if (FA and FB) else NAN
        sa, sb = set(AA), set(AB)
        inter = len(sa & sb)
        LMA, LMB = _sets(lma), _sets(lmb)
        sfa, sfb = set(FA), set(FB)
        lm = NAN
        if LMA or LMB:
            lm = max(len(LMA & sfb) / len(LMA) if LMA else 0.0, len(LMB & sfa) / len(LMB) if LMB else 0.0)
        UA, UB = ua.split(), ub.split()
        su, sv = set(UA), set(UB)
        PA, PB = _sets(pa), _sets(pb)
        if su and sv:
            nsh = len(su & sv)
            num = (nsh / len(su | sv), float(UA[0] == UB[0]), float(nsh == 0), float(nsh))
        else:
            num = (NAN, NAN, NAN, 0.0)
        if PA and PB:
            post = (float(bool(PA & PB)), float(not (PA & PB)), 0.0)
        else:
            post = (0.0, 0.0, float(bool(PA) != bool(PB)))
        sA, sB = set(A), set(B)
        out[i] = (
            *sd, var_sd, var_sd - sd[0],
            float(bool(LA) and LA == LB), float(bool(LA) and bool(LB) and not (LA & LB)),
            float(bool(LA) != bool(LB)),
            float(acronym_match(A, B)),
            len(SA & SB) / len(SA | SB) if (SA or SB) else NAN,
            float(bool(NA_ & NB_)) if (NA_ and NB_) else NAN,
            float(not (NA_ & NB_)) if (NA_ and NB_) else NAN,
            float(A[0] == B[0]) if (A and B) else NAN,
            tok_sim(A[0], B[0]) if (A and B) else NAN,
            tok_sim(A[-1], B[-1]) if (A and B) else NAN,
            asd[0], asd[1], asd[2], asd[3], asd[4], asd[5], afsd,
            inter / min(len(sa), len(sb)) if (sa and sb) else NAN,
            inter / len(sa) if (sa and sb) else NAN,
            inter / len(sb) if (sa and sb) else NAN,
            lm, *num, *post,
            len(sA & sfb) / len(sA) if (sA and sfb) else NAN,
            len(sB & sfa) / len(sB) if (sB and sfa) else NAN,
            len(A), len(B), len(AA), len(AB), float(not AA), float(not AB),
        )
    return out


def _field(n, key):
    if key == "nc":
        return n["n_core"].where(n["n_core"].str.len() > 0, n["n_full"]).to_numpy(object)
    if key == "nf":
        return n["n_full"].to_numpy(object)
    if key == "nr":
        return n["_nraw"].to_numpy(object)
    if key == "ac":
        return n["a_core"].where(n["a_core"].str.len() > 0, n["a_full"]).to_numpy(object)
    if key == "af":
        return n["a_full"].to_numpy(object)
    if key == "jt":
        return (n["n_full"] + " " + n["a_full"]).to_numpy(object)
    raise KeyError(key)


NEED_COLS = ["id", "src", "cty", "n_core", "n_full", "n_legal", "n_vars", "n_skel", "n_nums", "a_core", "a_full",
             "a_lm", "a_nums", "a_postal", "raw_name"]


def partition_stats(n1, npool):
    """Label-free frequency tables over a whole split partition, computed once and shared by every chunk
    (chains: how often a name / an address occurs among pool records and among S1 records)."""
    keyp = npool["cty"].astype(str) + "" + npool["n_core"].astype(str)
    key1 = n1["cty"].astype(str) + "" + n1["n_core"].astype(str)
    ap = npool["a_core"].astype(str)
    a1 = n1["a_core"].astype(str)
    akp = (npool["cty"].astype(str) + "" + ap)[ap != ""]
    ak1 = (n1["cty"].astype(str) + "" + a1)[a1 != ""]
    return {"cnt_pool": keyp.value_counts(), "cnt_s1": key1.value_counts(),
            "acnt": pd.concat([akp, ak1], ignore_index=True).value_counts()}


def _gather(n, idx):
    """Rows of n for the given positions (only the columns the features read), as a fresh 0..k-1 frame."""
    g = n.iloc[idx][[c for c in NEED_COLS if c in n.columns]].reset_index(drop=True)
    for c in g.columns:
        if c != "id":
            g[c] = g[c].astype(object).where(g[c].notna(), "")
    g["_nraw"] = [" ".join(str(x).lower().split()) for x in g["raw_name"]]   # raw name, casefolded
    return g


def pair_features(pairs: pd.DataFrame, n1: pd.DataFrame, npool: pd.DataFrame, idf: dict, n_jobs=-1, stats=None):
    """pairs: s1, cand (positions in n1 / npool), part, channel provenance, exact cosines, meta.
    Only the rows the pairs reference are touched, so this runs per chunk against multi-million-row pools.
    stats: partition_stats(n1_all, npool_all) for the rarity features (computed here if omitted).
    Returns a float32 DataFrame of features aligned with pairs."""
    t0 = time.time()
    workers = effective_cpus() if n_jobs in (-1, None) else n_jobs
    i1, ip = pairs["s1"].to_numpy(), pairs["cand"].to_numpy()
    g1, gp = _gather(n1, i1), _gather(npool, ip)
    stats = stats or partition_stats(n1, npool)
    F = {}
    # ---- vectorised C++ string metrics
    cache = {}
    for name, fld, scorer, scale in RF_SPECS:
        if fld not in cache:
            cache[fld] = (_field(g1, fld).tolist(), _field(gp, fld).tolist())
        a, b = cache[fld]
        v = cpdist(a, b, scorer, workers) * np.float32(scale)
        F[name] = v.astype(np.float32)
    ea = np.array([not x for x in cache["ac"][0]], dtype=bool)
    eb = np.array([not x for x in cache["ac"][1]], dtype=bool)
    for name, fld, _, _ in RF_SPECS:
        if fld in ("ac", "af"):
            F[name][ea | eb] = np.nan
    del cache
    # ---- python features (alignment etc.), parallel over chunks grouped by partition
    cols1 = ["n_core", "n_vars", "n_legal", "n_skel", "n_nums", "a_core", "a_full", "a_lm", "a_nums", "a_postal"]
    a1 = {c: g1[c].to_numpy(object) for c in cols1}
    ap = {c: gp[c].to_numpy(object) for c in cols1}
    part = pairs["part"].to_numpy(object)
    rows = list(zip(part, a1["n_core"], ap["n_core"], a1["n_vars"], ap["n_vars"], a1["n_legal"], ap["n_legal"],
                    a1["n_skel"], ap["n_skel"], a1["n_nums"], ap["n_nums"], a1["a_core"], ap["a_core"],
                    a1["a_full"], ap["a_full"], a1["a_lm"], ap["a_lm"], a1["a_nums"], ap["a_nums"],
                    a1["a_postal"], ap["a_postal"]))
    order = np.argsort(part, kind="stable")
    step = 40000
    chunks = [order[s:s + step] for s in range(0, len(order), step)]
    if len(chunks) > 1 and workers > 1:
        res = Parallel(n_jobs=workers)(
            delayed(_py_chunk)([rows[j] for j in ch], {k: idf[k] for k in set(part[ch])}) for ch in chunks)
    else:
        res = [_py_chunk([rows[j] for j in ch], idf) for ch in chunks]
    P = np.full((len(rows), len(PY_FEATS)), np.nan, np.float32)
    for ch, r in zip(chunks, res):
        P[ch] = r
    for j, name in enumerate(PY_FEATS):
        F[name] = P[:, j]
    del rows, P, res
    # ---- exact cosines + provenance from blocking / meta-blocking
    for _, col in EXACT:
        F[col] = pairs[col].to_numpy(np.float32)
    for ch in CHANNELS:
        F[f"s_{ch}"] = pairs[f"s_{ch}"].to_numpy(np.float32)
        F[f"k_{ch}"] = pairs[f"k_{ch}"].to_numpy(np.float32)
    F["n_ch"] = pairs["n_ch"].to_numpy(np.float32)
    F["meta"] = pairs["meta"].to_numpy(np.float32) if "meta" in pairs else np.zeros(len(pairs), np.float32)
    # ---- rarity / chains (label-free statistics of the whole partition, shared by all chunks)
    ka = pd.Series(g1["cty"].to_numpy(object) + "" + g1["n_core"].to_numpy(object))
    kb = pd.Series(gp["cty"].to_numpy(object) + "" + gp["n_core"].to_numpy(object))
    F["n_freq_pool_a"] = np.log1p(ka.map(stats["cnt_pool"]).fillna(0).to_numpy(np.float32))
    F["n_freq_pool_b"] = np.log1p(kb.map(stats["cnt_pool"]).fillna(0).to_numpy(np.float32))
    F["n_freq_s1_a"] = np.log1p(ka.map(stats["cnt_s1"]).fillna(0).to_numpy(np.float32))
    F["n_freq_s1_b"] = np.log1p(kb.map(stats["cnt_s1"]).fillna(0).to_numpy(np.float32))
    aka = pd.Series(g1["cty"].to_numpy(object) + "" + g1["a_core"].to_numpy(object))
    akb = pd.Series(gp["cty"].to_numpy(object) + "" + gp["a_core"].to_numpy(object))
    F["a_freq_a"] = np.log1p(aka.map(stats["acnt"]).fillna(0).to_numpy(np.float32))
    F["a_freq_b"] = np.log1p(akb.map(stats["acnt"]).fillna(0).to_numpy(np.float32))
    # ---- record flags and candidate-set structure (per S1: complete within any chunk of whole S1s)
    F["is_s3"] = (gp["src"].to_numpy(object) == "S3").astype(np.float32)
    F["same_cty"] = (g1["cty"].to_numpy(object) == gp["cty"].to_numpy(object)).astype(np.float32)
    s1c = pd.Series(i1)
    F["n_cands"] = s1c.map(s1c.value_counts()).to_numpy(np.float32)
    X = pd.DataFrame(F)
    log.info("features: %d pairs x %d features in %.1fs", len(X), X.shape[1], time.time() - t0)
    return X


# Monotone constraints (plan 7.2): +1 on core similarities, -1 on conflict features.
MONO_UP = ["n_sd", "n_cov_a", "n_cov_b", "n_cos_c", "j_cos_c", "a_cos_c", "n_tset"]
MONO_DOWN = ["n_mu_a", "n_mu_b", "post_conf2", "num_conf"]


def monotone_vector(cols):
    return [1 if c in MONO_UP else (-1 if c in MONO_DOWN else 0) for c in cols]
