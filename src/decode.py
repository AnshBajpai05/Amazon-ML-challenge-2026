"""Decision layer (plan section 8): exclusivity + per-S1 set decoding, tuned by nested CV.

D1: threshold policy (top-1 if p >= t1; extras if p >= t2 and p >= r * p_top).
D2/D3: exact expected-F0.5 over top-k sets under independence (Poisson-binomial DP),
       optionally with the entity model (E[F(empty)] = 1 - h) and a logit recalibration
       p' = sigmoid(a * logit p + b). D2 is D3 with a=1, b=0.
Everything is vectorised over entities: one (N, C) matrix of sorted candidate probabilities.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from .blocking import group_rank
from .models import fit_isotonic

log = logging.getLogger("ber")


def build_matrix(s1, p, n_s1, max_c, tie=None, y=None):
    """Sorted top-max_c candidates per S1: P (N,C) probs, IDX (N,C) pair row or -1, L labels, NV counts."""
    s1 = np.asarray(s1)
    p = np.asarray(p, np.float64)
    keys = (np.zeros(len(p)) if tie is None else -np.asarray(tie, np.float64), -p, s1)
    order = np.lexsort(keys)
    ss = s1[order]
    new = np.r_[True, ss[1:] != ss[:-1]] if len(ss) else np.zeros(0, bool)
    start = np.maximum.accumulate(np.where(new, np.arange(len(ss)), 0)) if len(ss) else np.zeros(0, int)
    rank = np.arange(len(ss)) - start
    keep = rank < max_c
    P = np.zeros((n_s1, max_c))
    IDX = np.full((n_s1, max_c), -1, np.int64)
    P[ss[keep], rank[keep]] = p[order][keep]
    IDX[ss[keep], rank[keep]] = order[keep]
    L = None
    if y is not None:
        L = np.zeros((n_s1, max_c), np.float32)
        L[ss[keep], rank[keep]] = np.asarray(y, np.float32)[order][keep]
    NV = np.minimum(np.bincount(s1, minlength=n_s1), max_c) if len(s1) else np.zeros(n_s1, int)
    return P, IDX, L, NV


def expected_f_best_k(P, NV, h=None, chunk=20000):
    """Best k (top-k set) per row maximising E[F0.5]. Exact under independence."""
    vals, p0 = expected_f_values(P, NV, chunk)
    return best_k_from(vals, p0, h)


def best_k_from(vals, p0, h=None):
    """vals[:, k] = E[F0.5 | pick top-k] (no entity model); with h, E[F(empty)] = 1 - h and the
    non-empty values are rescaled from P(>=1 true) = 1 - p0 to h."""
    if h is None:
        return np.argmax(vals, axis=1)
    v = vals.copy()
    den = 1.0 - p0
    scale = np.where(den > 1e-9, h / np.maximum(den, 1e-12), 0.0)
    fin = np.isfinite(v)
    fin[:, 0] = False
    v = np.where(fin, np.where(fin, v, 0.0) * scale[:, None], v)
    v[:, 0] = 1.0 - h
    return np.argmax(v, axis=1)


def _dp_block(p, C):
    """Exact E[F0.5] of every top-k set (k = 0..C) for rows of p (n, C), independence assumed."""
    n = len(p)
    T = np.zeros((C + 1, n, C + 1))
    T[0, :, 0] = 1.0
    for k in range(1, C + 1):
        pk = p[:, k - 1:k]
        T[k] = T[k - 1] * (1 - pk)
        T[k][:, 1:] += T[k - 1][:, :-1] * pk
    U = np.zeros((C + 1, n, C + 1))
    U[C, :, 0] = 1.0
    for k in range(C - 1, -1, -1):
        pk = p[:, k:k + 1]
        U[k] = U[k + 1] * (1 - pk)
        U[k][:, 1:] += U[k + 1][:, :-1] * pk
    vals = np.zeros((n, C + 1))
    vals[:, 0] = T[C][:, 0]
    for k in range(1, C + 1):
        t = np.arange(k + 1)[:, None]
        u = np.arange(C - k + 1)[None, :]
        W = 1.25 * t / (k + 0.25 * (t + u))
        vals[:, k] = ((T[k][:, :k + 1] @ W) * U[k][:, :C - k + 1]).sum(axis=1)
    return vals, T[C][:, 0]


def expected_f_values(P, NV, chunk=20000, min_p=1e-3):
    """(vals (N, C+1), p0 (N,)) with vals[:, k] = E[F0.5] of the top-k set, -inf where k > #candidates.
    Candidates with p < min_p are dropped (as in plan 8.2); rows are processed in groups of equal effective
    candidate count, so most entities (1-3 plausible candidates) cost a tiny DP."""
    N, C = P.shape
    vals_all = np.full((N, C + 1), -np.inf)
    p0_all = np.ones(N)
    nv = np.minimum(NV, (P >= min_p).sum(axis=1))
    vals_all[:, 0] = 1.0
    for c in np.unique(nv):
        rows = np.flatnonzero(nv == c)
        if c == 0:
            continue
        for s in range(0, len(rows), chunk):
            r = rows[s:s + chunk]
            v, p0 = _dp_block(P[r, :c], int(c))
            vals_all[r, :c + 1] = v
            p0_all[r] = p0
    return vals_all, p0_all


def d1_k(P, NV, t1, t2, r):
    top = P[:, 0]
    C = P.shape[1]
    take = (top >= t1) & (NV >= 1)
    ext = (P[:, 1:] >= t2) & (P[:, 1:] >= r * top[:, None]) & (np.arange(1, C)[None, :] < NV[:, None])
    # P is sorted, so both conditions hold on a prefix; cumprod enforces it anyway
    n_ext = np.cumprod(ext, axis=1).sum(axis=1)
    return np.where(take, 1 + n_ext, 0)


def f_scores(k, L, G):
    """Per-S1 F0.5 for top-k picks. L: labels in sorted order, G: gold count (incl. non-candidates)."""
    N = len(k)
    cum = np.concatenate([np.zeros((N, 1)), np.cumsum(L, axis=1)], axis=1)
    tp = cum[np.arange(N), k]
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(tp > 0, 1.25 * tp / (k + 0.25 * G), 0.0)
    return np.where(G > 0, f, (k == 0).astype(np.float64))


def recal(P, a, b):
    if a == 1.0 and b == 0.0:
        return P
    q = np.clip(P, 1e-7, 1 - 1e-7)
    out = 1.0 / (1.0 + np.exp(-(a * np.log(q / (1 - q)) + b)))
    return np.where(P > 0, out, 0.0)


def exclusivity(p, rank_in_cand, gamma):
    """S1 is deduplicated, so a pool record belongs to at most one S1: damp its weaker claims."""
    return np.where(rank_in_cand > 1, p * gamma, p)


def _configs(dc):
    d3 = [("d3", g, a, b, uh) for g in dc["gammas"] for a in dc["recal_a"] for b in dc["recal_b"]
          for uh in ((False, True) if dc.get("use_h", True) else (False,))]
    d1 = [("d1", g, t1, t2, r) for g in dc["gammas"] for t1 in dc["d1_t1"] for t2 in dc["d1_t2"]
          for r in dc["d1_r"]]
    return d3 + d1


def _eval_all(s1, p_cal, p_raw, rank_c, h_cal, y, G, n_s1, dc, configs, s1_fold, n_folds):
    """F0.5 summed per fold for every config -> dict config -> array(n_folds).
    Only per-fold sums are kept (nested CV needs nothing else), so memory stays O(#configs x #folds)."""
    out = {}
    C = dc["max_c"]
    by_g = {}
    for c in configs:
        by_g.setdefault(c[1], []).append(c)
    ar = np.arange(n_s1)
    pos = G > 0

    for g, cs in by_g.items():
        pe = exclusivity(p_cal, rank_c, g)
        P, IDX, L, NV = build_matrix(s1, pe, n_s1, C, tie=p_raw, y=y)
        cum = np.concatenate([np.zeros((n_s1, 1)), np.cumsum(L, axis=1)], axis=1)   # TP of top-k, once per gamma

        def fold_sums(k):
            tp = cum[ar, k]
            with np.errstate(divide="ignore", invalid="ignore"):
                f = np.where(pos, np.where(tp > 0, 1.25 * tp / (k + 0.25 * G), 0.0), (k == 0).astype(np.float64))
            return np.bincount(s1_fold, weights=f, minlength=n_folds)

        dp, ext = {}, {}
        top = P[:, 0]
        has = NV >= 1
        valid = np.arange(1, C)[None, :] < NV[:, None]
        for c in cs:
            if c[0] == "d3":
                _, _, a, b, uh = c
                if (a, b) not in dp:
                    dp[(a, b)] = expected_f_values(recal(P, a, b), NV)
                k = best_k_from(*dp[(a, b)], h_cal if uh else None)
            else:
                _, _, t1, t2, r = c
                if (t2, r) not in ext:                      # extras depend on (t2, r) only; reuse across t1
                    e = (P[:, 1:] >= t2) & (P[:, 1:] >= r * top[:, None]) & valid
                    ext[(t2, r)] = np.cumprod(e, axis=1).sum(axis=1)
                k = np.where((top >= t1) & has, 1 + ext[(t2, r)], 0)
            out[c] = fold_sums(k)
    return out


def _argbest(configs, S, fold_mask):
    """Highest mean F over the folds in fold_mask; ties go to the earlier (simpler) config in the grid order."""
    best, best_v = configs[0], -1.0
    for c in configs:
        v = float(S[c][fold_mask].sum())
        if v > best_v + 1e-12:
            best, best_v = c, v
    return best


def tune_decoder(s1, cand, p_raw, y, h_raw, h_label, s1_fold, G, n_s1, dc):
    """Nested CV over decoder configs. Returns report dict + fitted 'all' calibrators and best config."""
    t0 = time.time()
    s1 = np.asarray(s1)
    rank_c = group_rank(cand, p_raw, tie=s1)
    configs = _configs(dc)
    s1_fold = np.asarray(s1_fold, np.int64)
    n_folds = int(s1_fold.max()) + 1 if n_s1 else 1
    fold_n = np.bincount(s1_fold, minlength=n_folds).astype(np.float64)
    nested_sum = 0.0
    per_fold = []
    for f in np.unique(s1_fold):
        trp = s1_fold[s1] != f
        iso = fit_isotonic(p_raw[trp], y[trp])
        tre = s1_fold != f
        iso_h = fit_isotonic(h_raw[tre], h_label[tre])
        S = _eval_all(s1, iso.predict(p_raw), p_raw, rank_c, iso_h.predict(h_raw), y, G, n_s1, dc, configs,
                      s1_fold, n_folds)
        best = _argbest(configs, S, np.arange(n_folds) != f)
        nested_sum += S[best][f]
        per_fold.append({"fold": int(f), "config": list(best), "f05": float(S[best][f] / max(fold_n[f], 1))})
    iso = fit_isotonic(p_raw, y)
    iso_h = fit_isotonic(h_raw, h_label)
    S = _eval_all(s1, iso.predict(p_raw), p_raw, rank_c, iso_h.predict(h_raw), y, G, n_s1, dc, configs,
                  s1_fold, n_folds)
    best = _argbest(configs, S, np.ones(n_folds, bool))
    mean = {c: float(v.sum() / max(n_s1, 1)) for c, v in S.items()}
    table = sorted(((m, list(c)) for c, m in mean.items()), key=lambda x: -x[0])

    def best_of(pred):
        cs = [c for c in configs if pred(c)]
        if not cs:
            return {"config": None, "f05_in_sample": float("nan")}
        c = max(cs, key=lambda c: mean[c])
        return {"config": list(c), "f05_in_sample": mean[c]}

    ablation = {
        "global_threshold (d1, r=0, t1=t2, no exclusivity)": best_of(
            lambda c: c[0] == "d1" and c[1] == 1.0 and c[4] == 0.0 and c[2] == c[3]),
        "d1 threshold policy": best_of(lambda c: c[0] == "d1"),
        "d2 expected-F0.5 (a=1,b=0, no h, no exclusivity)": best_of(
            lambda c: c[0] == "d3" and c[1] == 1.0 and c[2] == 1.0 and c[3] == 0.0 and not c[4]),
        "d2 + entity model h": best_of(
            lambda c: c[0] == "d3" and c[1] == 1.0 and c[2] == 1.0 and c[3] == 0.0 and c[4]),
        "d3 recalibrated + h + exclusivity (full grid)": best_of(lambda c: c[0] == "d3"),
    }
    rep = {"nested_f05": float(nested_sum / max(n_s1, 1)), "per_fold": per_fold, "best_config": list(best),
           "best_in_sample_f05": mean[best], "top10": table[:10], "decoder_ablation": ablation,
           "seconds": round(time.time() - t0, 1)}
    log.info("decoder: nested F0.5=%.5f  best=%s (in-sample %.5f)  %.1fs", rep["nested_f05"], best,
             rep["best_in_sample_f05"], time.time() - t0)
    return rep, {"iso": iso, "iso_h": iso_h, "config": best, "grid": table}


def decode(s1, cand, p_raw, h_raw, n_s1, dc, fitted):
    """Apply the tuned policy. Returns (k per S1, IDX matrix, calibrated sorted P) for selection."""
    s1 = np.asarray(s1)
    rank_c = group_rank(cand, p_raw, tie=s1)
    cfg_ = tuple(fitted["config"])
    p_cal = fitted["iso"].predict(p_raw) if len(p_raw) else np.zeros(0)
    h_cal = fitted["iso_h"].predict(h_raw)
    P, IDX, _, NV = build_matrix(s1, exclusivity(p_cal, rank_c, cfg_[1]), n_s1, dc["max_c"], tie=p_raw)
    if cfg_[0] == "d3":
        _, _, a, b, uh = cfg_
        k = expected_f_best_k(recal(P, a, b), NV, h_cal if uh else None)
    else:
        _, _, t1, t2, r = cfg_
        k = d1_k(P, NV, t1, t2, r)
    return k, IDX, P


def selections(k, IDX):
    """Row indices (into the pairs table) of the chosen candidates for each S1."""
    return [IDX[i, :k[i]].tolist() for i in range(len(k))]


