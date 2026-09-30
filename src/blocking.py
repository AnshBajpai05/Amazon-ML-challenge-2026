"""Stage 1: multi-channel blocking + learned meta-blocking (plan section 5, A.5).

Per partition (country label by default), TF-IDF vocabularies and IDF are fitted on
S1 u pool of the split being processed (label-free), so unseen countries such as France
get sensible weights. Channels:
  c1 name char 3-4g | c2 name word | c3 name skeleton | c4 address char | c5 name+address char
  c6 reverse kNN (pool -> S1 on c5) | c7 2-hop (pool near-duplicates of each S1's top c5 hits)
  c9 reverse name kNN (pool -> S1 on name only; finds address-less records in crowded name neighbourhoods)
  c8 dense embeddings (optional, GPU)
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .compat import effective_cpus

try:
    from sparse_dot_topn import sp_matmul_topn
except Exception:  # optional fast path
    sp_matmul_topn = None

log = logging.getLogger("ber")

CHANNELS = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"]
EXACT = [("c1", "n_cos_c"), ("c2", "n_cos_w"), ("c3", "n_cos_s"), ("c4", "a_cos_c"), ("aw", "a_cos_w"),
         ("c5", "j_cos_c")]
META_FEATS = ([f"s_{c}" for c in CHANNELS] + [f"k_{c}" for c in CHANNELS] +
              ["n_ch"] + [c for _, c in EXACT] + ["post_eq", "post_conf"])


def _threads():
    return effective_cpus()


def texts(n: pd.DataFrame) -> dict:
    name = n["n_core"].where(n["n_core"].str.len() > 0, n["n_full"]).to_numpy(object)
    addr = n["a_core"].where(n["a_core"].str.len() > 0, n["a_full"]).to_numpy(object)
    joint = np.array([f"{a} | {b}" for a, b in zip(name, addr)], dtype=object)
    return {"name": name, "skel": n["n_skel"].to_numpy(object), "addr": addr, "joint": joint}


def _vectorizer(kind, n_docs, max_df):
    md = max_df if (max_df < 1.0 and n_docs >= 2000) else 1.0
    if kind == "char":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), sublinear_tf=True, max_df=md,
                               lowercase=False, dtype=np.float32)
    return TfidfVectorizer(analyzer="word", token_pattern=r"\S+", sublinear_tf=True, max_df=md,
                           lowercase=False, dtype=np.float32)


def fit_pair(kind, a, b, max_df=1.0):
    """Fit on a u b (no labels), return (vectorizer, A, B) or None when nothing survives."""
    vec = _vectorizer(kind, len(a) + len(b), max_df)
    try:
        vec.fit(np.concatenate([a, b]))
    except ValueError:  # empty vocabulary (e.g. all names empty)
        return None
    return vec, vec.transform(a).tocsr(), vec.transform(b).tocsr()


def _empty():
    z = np.zeros(0, np.int64)
    return z, z.copy(), np.zeros(0, np.float32), np.zeros(0, np.float32)


def group_rank(keys, vals, tie=None):
    """1-based rank of vals (descending) within each key group; deterministic tie-break."""
    keys = np.asarray(keys)
    vals = np.asarray(vals)
    if len(keys) == 0:
        return np.zeros(0, np.float32)
    order = np.lexsort((tie,) + (-vals, keys) if tie is not None else (-vals, keys))
    ks = keys[order]
    new = np.r_[True, ks[1:] != ks[:-1]]
    start = np.maximum.accumulate(np.where(new, np.arange(len(ks)), 0))
    rank = np.empty(len(keys), np.float32)
    rank[order] = np.arange(len(ks)) - start + 1
    return rank


def sparse_topk(A, B, k, min_sim=0.0, budget=int(3e7), BT=None):
    """Top-k cosine neighbours of every row of A among rows of B (L2-normalised CSR).
    Returns (rows_in_A, rows_in_B, scores, rank). Never builds a dense |A| x |B| matrix.
    BT: optional precomputed B.T as float32 CSR (reused across many query chunks)."""
    if k <= 0 or A.shape[0] == 0 or B.shape[0] == 0 or A.nnz == 0 or B.nnz == 0:
        return _empty()
    A = A.astype(np.float32)
    if BT is None:
        BT = B.T.tocsr().astype(np.float32)
    if sp_matmul_topn is not None:
        M = sp_matmul_topn(A, BT, top_n=int(k), threshold=(min_sim if min_sim > 0 else None), sort=True,
                           n_threads=_threads()).tocsr()
        rows = np.repeat(np.arange(M.shape[0]), np.diff(M.indptr))
        cols, sc = M.indices.astype(np.int64), M.data.astype(np.float32)
    else:
        R, C, S = [], [], []
        s, chunk = 0, 256
        while s < A.shape[0]:
            M = (A[s:s + chunk] @ BT).tocsr()
            nxt = max(16, min(16384, int(chunk * budget / max(M.nnz, 1))))
            if min_sim > 0:
                M.data[M.data < min_sim] = 0
                M.eliminate_zeros()
            cnt = np.diff(M.indptr)
            rr = np.repeat(np.arange(M.shape[0]), cnt)
            order = np.lexsort((M.indices, -M.data, rr))
            rank = np.arange(len(order)) - np.repeat(M.indptr[:-1], cnt)
            sel = order[rank < k]
            R.append(rr[sel] + s)
            C.append(M.indices[sel])
            S.append(M.data[sel])
            s += chunk
            chunk = nxt
        rows, cols, sc = np.concatenate(R), np.concatenate(C).astype(np.int64), np.concatenate(S)
    keep = sc > max(min_sim, 1e-9)
    rows, cols, sc = rows[keep].astype(np.int64), cols[keep], sc[keep].astype(np.float32)
    return rows, cols, sc, group_rank(rows, sc, tie=cols)


def rowwise_cos(A, B, ia, ib, chunk=400_000):
    out = np.zeros(len(ia), np.float32)
    for s in range(0, len(ia), chunk):
        x = A[ia[s:s + chunk]]
        y = B[ib[s:s + chunk]]
        out[s:s + chunk] = np.asarray(x.multiply(y).sum(axis=1)).ravel()
    return out


def _topk_split(A, B, k, k_deep, deep_rows):
    """sparse_topk with a larger k for the rows flagged in deep_rows (e.g. S1 records without an address)."""
    if k_deep <= k or not deep_rows.any():
        return sparse_topk(A, B, k)
    out = []
    for mask, kk in ((~deep_rows, k), (deep_rows, k_deep)):
        idx = np.flatnonzero(mask)
        if len(idx):
            r, c, s, rk = sparse_topk(A[idx], B, kk)
            out.append((idx[r], c, s, rk))
    if not out:
        return _empty()
    return tuple(np.concatenate([o[i] for o in out]) for i in range(4))


def block_partition(n1, npo, cfg, dense=None):
    """Blocking inside one partition. n1/npo are normalized frames (local positions).
    Returns (pairs with local r/c, idf maps, per-channel pair counts)."""
    K = cfg["k"]
    T1, T2 = texts(n1), texts(npo)
    spec = {"c1": ("char", "name", cfg["max_df_char"]), "c2": ("word", "name", 1.0),
            "c3": ("word", "skel", 1.0), "c4": ("char", "addr", cfg["max_df_char"]),
            "c5": ("char", "joint", cfg["max_df_joint"]), "aw": ("word", "addr", 1.0)}
    M = {key: fit_pair(kind, T1[f], T2[f], md) for key, (kind, f, md) in spec.items()}
    parts = []
    # records without an address can only be found through their name: search the name channels deeper
    no_addr = np.array([not a.strip() for a in T1["addr"]], dtype=bool)
    mult = float(cfg.get("empty_addr_k_mult", 2.0))
    for ch in ("c1", "c2", "c3", "c4", "c5"):
        if K.get(ch, 0) > 0 and M[ch] is not None:
            deep = int(K[ch] * mult) if ch in ("c1", "c2", "c3") else K[ch]
            parts.append((ch, *_topk_split(M[ch][1], M[ch][2], K[ch], deep, no_addr)))
    if K.get("c9", 0) > 0 and M["c1"] is not None:             # reverse name kNN: pool -> S1 on name only
        r, c, s, rk = sparse_topk(M["c1"][2], M["c1"][1], K["c9"])
        parts.append(("c9", c, r, s, rk))
    if M["c5"] is not None:
        A5, B5 = M["c5"][1], M["c5"][2]
        if K.get("c6", 0) > 0:                                  # reverse kNN: pool -> S1
            r, c, s, rk = sparse_topk(B5, A5, K["c6"])
            parts.append(("c6", c, r, s, rk))
        c5 = next((p for p in parts if p[0] == "c5"), None)
        if K.get("c7", 0) > 0 and c5 is not None and len(c5[1]):  # 2-hop through pool near-duplicates
            _, r5, c5c, s5, k5 = c5
            m = (k5 <= cfg["twohop_seed_k"]) & (s5 >= cfg["twohop_seed_min"])
            seeds = np.unique(c5c[m])
            if len(seeds):
                rr, cc, ss, _ = sparse_topk(B5[seeds], B5, cfg["twohop_nb_k"] + 1, min_sim=cfg["twohop_min_sim"])
                seed_of = seeds[rr]
                ok = cc != seed_of
                nb = pd.DataFrame({"seed": seed_of[ok], "c": cc[ok], "sim": ss[ok]})
                sd = pd.DataFrame({"r": r5[m], "seed": c5c[m], "ss": s5[m]})
                j = sd.merge(nb, on="seed")
                if len(j):
                    j["score"] = (j["sim"] * j["ss"]).astype(np.float32)
                    j = j.groupby(["r", "c"], as_index=False)["score"].max()
                    rk = group_rank(j["r"].to_numpy(), j["score"].to_numpy(), tie=j["c"].to_numpy())
                    j = j[rk <= K["c7"]]
                    parts.append(("c7", j["r"].to_numpy(np.int64), j["c"].to_numpy(np.int64),
                                  j["score"].to_numpy(np.float32), rk[rk <= K["c7"]]))
    if dense is not None and K.get("c8", 0) > 0:
        from .neural import dense_topk
        E1, E2 = dense
        r, c, s = dense_topk(E1, E2, K["c8"])
        parts.append(("c8", r, c, s, group_rank(r, s, tie=c)))

    n2 = len(npo)
    if not parts:
        return None, _idf(M), {}
    keys = np.unique(np.concatenate([p[1] * n2 + p[2] for p in parts]))
    pairs = pd.DataFrame({"r": keys // n2, "c": keys % n2})
    n_ch = np.zeros(len(keys), np.int8)
    counts = {}
    for ch in CHANNELS:
        pairs[f"s_{ch}"] = np.float32(0.0)
        pairs[f"k_{ch}"] = np.float32(K.get(ch, 0) + 1)
    for ch, r, c, s, rk in parts:
        pos = np.searchsorted(keys, r * n2 + c)
        sc = pairs[f"s_{ch}"].to_numpy().copy()
        kk = pairs[f"k_{ch}"].to_numpy().copy()
        np.maximum.at(sc, pos, s)
        kk[pos] = np.minimum(kk[pos], rk)
        pairs[f"s_{ch}"] = sc
        pairs[f"k_{ch}"] = kk
        hit = np.zeros(len(keys), bool)
        hit[pos] = True
        n_ch += hit
        counts[ch] = int(hit.sum())
    pairs["n_ch"] = n_ch
    ia, ib = pairs["r"].to_numpy(), pairs["c"].to_numpy()
    for key, col in EXACT:
        pairs[col] = rowwise_cos(M[key][1], M[key][2], ia, ib) if M[key] is not None else np.float32(0.0)
    return pairs, _idf(M), counts


def _idf(M):
    def one(key):
        if M.get(key) is None:
            return {}, 8.0
        v = M[key][0]
        idf = dict(zip(v.get_feature_names_out().tolist(), v.idf_.astype(float).tolist()))
        return idf, float(v.idf_.max()) if len(v.idf_) else 8.0
    (ni, nd), (ai, ad) = one("c2"), one("aw")
    return ni, ai, nd, ad


def partition_keys(n1, npool, mode):
    if mode == "none":
        return np.full(len(n1), "__all__", object), np.full(len(npool), "__all__", object)
    return n1["cty"].to_numpy(object), npool["cty"].to_numpy(object)


def run_blocking(n1, npool, cfg, mode="country", dense=None):
    """Blocking over every partition. Returns (pairs with global s1/cand indices, idf per partition, stats)."""
    t0 = time.time()
    k1, kp = partition_keys(n1, npool, mode)
    out, idf, stats = [], {}, {}
    for key in sorted(set(k1)):
        i1 = np.flatnonzero(k1 == key)
        ip = np.flatnonzero(kp == key)
        if len(ip) == 0:
            stats[key] = {"s1": int(len(i1)), "pool": 0, "pairs": 0}
            continue
        d = None
        if dense is not None:
            d = (dense[0][i1], dense[1][ip])
        pairs, idf_k, counts = block_partition(n1.iloc[i1].reset_index(drop=True),
                                               npool.iloc[ip].reset_index(drop=True), cfg["blocking"], d)
        idf[key] = idf_k
        if pairs is None:
            stats[key] = {"s1": int(len(i1)), "pool": int(len(ip)), "pairs": 0}
            continue
        pairs.insert(0, "s1", i1[pairs.pop("r").to_numpy()])
        pairs.insert(1, "cand", ip[pairs.pop("c").to_numpy()])
        pairs["part"] = key
        out.append(pairs)
        stats[key] = {"s1": int(len(i1)), "pool": int(len(ip)), "pairs": int(len(pairs)), "channel_pairs": counts}
        log.info("  block %-12s S1=%7d pool=%8d union=%9d (%.1f/S1)", key[:12], len(i1), len(ip), len(pairs),
                 len(pairs) / max(len(i1), 1))
    if out:
        pairs = pd.concat(out, ignore_index=True)
    else:
        pairs = pd.DataFrame(columns=["s1", "cand", "part"] + META_FEATS[:-2])
    pairs = pairs.sort_values(["s1", "cand"], kind="stable").reset_index(drop=True)
    add_postal(pairs, n1, npool)
    log.info("blocking done: %d union pairs in %.1fs", len(pairs), time.time() - t0)
    return pairs, idf, stats


def add_postal(pairs, n1, npool):
    """Cheap exact signal for the meta-blocker: any shared postal-like token / conflicting ones."""
    pa = n1["a_postal"].to_numpy(object)[pairs["s1"].to_numpy()]
    pb = npool["a_postal"].to_numpy(object)[pairs["cand"].to_numpy()]
    eq = np.zeros(len(pairs), np.float32)
    conf = np.zeros(len(pairs), np.float32)
    for i, (a, b) in enumerate(zip(pa, pb)):
        if a and b:
            if set(a.split()) & set(b.split()):
                eq[i] = 1.0
            else:
                conf[i] = 1.0
    pairs["post_eq"] = eq
    pairs["post_conf"] = conf


def choose_tau(s1, score, y, top_keep, cap, ratio):
    """Largest tau such that kept positives >= ratio x union positives (OOF scores)."""
    rank = group_rank(s1, score)
    y = np.asarray(y, bool)
    total = int(y.sum())
    if total == 0:
        return 1.0
    in_cap = rank <= cap
    base = int((y & in_cap & (rank <= top_keep)).sum())
    need = int(np.ceil(ratio * total)) - base
    if need <= 0:
        return 1.0 + 1e-6
    rest = np.sort(score[y & in_cap & (rank > top_keep)])[::-1]
    if need > len(rest):
        return 0.0
    return float(rest[need - 1])


def select_candidates(s1, score, top_keep, tau, cap):
    rank = group_rank(s1, score)
    return ((rank <= top_keep) | (score >= tau)) & (rank <= cap)
