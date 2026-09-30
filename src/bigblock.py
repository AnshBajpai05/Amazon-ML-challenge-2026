"""Blocking for multi-million-record partitions (the real challenge data: ~5M pool records per country).

Character n-grams are infeasible at this size (a common 3-gram has a posting list of hundreds of thousands
of records), so every channel is word-level: hashed TF-IDF over canonical tokens, plus a phonetic skeleton
token (~prvt) and a 4-character prefix token (^sunr) per word, which restore tolerance to typos and
transliteration variants. Hashing needs no vocabulary, so transforms run in parallel processes.

Channels (names kept compatible with blocking.CHANNELS / features.py):
  c2 name words | c4 address words | c5 name + address words      forward: each S1 chunk -> pool top-K
  c6 reverse joint | c9 reverse name                              reverse: every pool record -> S1 top-K
IDF comes from document frequencies over S1 u pool of the partition being searched (label-free).
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
from sklearn.utils import murmurhash3_32

from .blocking import CHANNELS, EXACT, _empty, group_rank, rowwise_cos, sparse_topk
from .compat import effective_cpus

log = logging.getLogger("ber")
N_FEATURES = 2 ** 22
_HV = HashingVectorizer(n_features=N_FEATURES, token_pattern=r"\S+", lowercase=False, alternate_sign=False,
                        norm=None, dtype=np.float32)


def _bigrams(toks):
    return [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


def _name_text(core, skel):
    """Words + phonetic skeletons + 4-char prefixes + adjacent-word bigrams. Business names combine very common
    words ('star solutions', 'sai traders'): each word alone matches tens of thousands of records and is pruned,
    the bigram is rare. On labelled train data (India, 20k S1) bigrams lift joint-channel recall 0.73 -> 0.93."""
    toks = core.split()
    return " ".join(toks + ["~" + k for k in skel.split()] + ["^" + t[:4] for t in toks if len(t) >= 6] +
                    _bigrams(toks))


def _addr_text(core):
    toks = core.split()
    return " ".join(toks + ["^" + t[:4] for t in toks if len(t) >= 6 and not t.isdigit()] + _bigrams(toks))


def _numloc_text(core, nums):
    """House number x locality keys ('28#chennai'): a native-script or website-style name shares no word with
    its S1, and an abbreviated address shares only common words (city, state); the pair of the house number and
    a locality word is rare and survives in both."""
    toks = list(dict.fromkeys(t for t in core.split() if len(t) >= 3 and not t.isdigit()))[-8:]  # locality = tail
    return " ".join(f"{n}#{t}" for n in nums.split()[:2] for t in toks)


def _keys_text(core):
    """Exact 'generator' keys shared by a name and its website / handle / initials / reordered forms:
    concatenated name (olaniq pet care = olaniqpetcare.com), sorted words (reordering), initials forms
    (stewart gustafson cooley -> sg / sgc; cameron cerrone fields -> ccfields) and the name without its first word
    (economic development institute -> developmentinstitute). Common keys are pruned by the posting cap."""
    t = [w for w in core.split() if not w.isdigit()]
    if not t:
        return ""
    keys = {"d_" + "".join(t), "s_" + "_".join(sorted(t))}
    if len(t) >= 2:
        keys |= {"d_" + "".join(w[0] for w in t[:2]), "d_" + "".join(w[0] for w in t[:-1]) + t[-1],
                 "d_" + "".join(t[1:]), "d_" + "".join(t[:2])}
    if len(t) >= 3:
        keys |= {"d_" + "".join(w[0] for w in t[:3]), "d_" + "".join(w[0] for w in t)}
    return " ".join(k for k in keys if len(k) >= 4)                  # 'd_' + >= 2 letters


def _texts(n):
    core = n["n_core"].astype(object).where(n["n_core"].astype(str).str.len() > 0, n["n_full"].astype(object))
    core = core.fillna("").to_numpy(object)
    skel = n["n_skel"].astype(object).fillna("").to_numpy(object)
    addr = n["a_core"].astype(object).where(n["a_core"].astype(str).str.len() > 0, n["a_full"].astype(object))
    addr = addr.fillna("").to_numpy(object)
    nums = n["a_nums"].astype(object).fillna("").to_numpy(object)
    name = [_name_text(c, k) for c, k in zip(core, skel)]
    adr = [_addr_text(a) for a in addr]
    joint = [f"{a} {b}" for a, b in zip(name, adr)]
    return {"c2": name, "c4": adr, "c5": joint, "c7": [_numloc_text(a, m) for a, m in zip(addr, nums)],
            "c1": [_keys_text(c) for c in core]}


def _hash_chunk(texts):
    X = _HV.transform(texts)
    X.data = np.log1p(X.data)                                    # sublinear tf
    return X.tocsr()


def hash_transform(texts, n_jobs=None):
    n = n_jobs or effective_cpus()
    if len(texts) < 50000 or n == 1:
        return _hash_chunk(texts)
    step = len(texts) // (n * 2) + 1
    parts = Parallel(n_jobs=n)(delayed(_hash_chunk)(texts[i:i + step]) for i in range(0, len(texts), step))
    return sp.vstack(parts).tocsr()


class HashedIdf:
    """dict-like token -> idf backed by the hashed document-frequency array (what soft_align needs)."""

    def __init__(self, idf, default):
        self.idf, self.default = idf, float(default)

    def get(self, tok, default=None):
        v = self.idf[murmurhash3_32(tok, positive=True) % N_FEATURES]
        return float(v) if v > 0 else (self.default if default is None else default)


def _weight(X, idf):
    X = X @ sp.diags(idf)
    X = normalize(X, copy=False).tocsr()
    X.eliminate_zeros()
    return X.astype(np.float32)


class PartitionIndex:
    """All S1 + pool records of one country partition of one split. Forward channels are queried per S1 chunk;
    reverse channels are computed once for every pool record."""

    def __init__(self, s1, pool, cfg):
        t0 = time.time()
        self.cfg = cfg
        self.K = cfg["k"]
        self.n_pool = len(pool)
        ts, tp = _texts(s1), _texts(pool)
        self.S, self.P, self.PT, self.idf_full = {}, {}, {}, {}
        N = len(s1) + len(pool)
        for ch in ("c2", "c4", "c5") + tuple(c for c in ("c7", "c1") if self.K.get(c, 0) > 0):
            A, B = hash_transform(ts[ch]), hash_transform(tp[ch])
            df = np.bincount(A.indices, minlength=N_FEATURES) + np.bincount(B.indices, minlength=N_FEATURES)
            idf = np.where(df > 0, np.log((1 + N) / (1 + df)) + 1, 0).astype(np.float32)
            self.idf_full[ch] = idf
            w = idf.copy()
            # prune common tokens (city names, 'pvt'): a posting list longer than the cap costs query time and
            # rarely identifies a match. The absolute cap keeps query cost flat as partitions grow.
            mda = cfg.get("max_df_abs") or 1e18
            cap_abs = mda.get(ch, 1e18) if isinstance(mda, dict) else mda
            w[df > min(cfg.get("max_df_word", 0.02) * N, cap_abs)] = 0.0
            self.S[ch], self.P[ch] = _weight(A, w), _weight(B, w)
            self.PT[ch] = self.P[ch].T.tocsr()
            del A, B, df
        default_n = float(self.idf_full["c2"].max() or 8.0)
        default_a = float(self.idf_full["c4"].max() or 8.0)
        self.idf = (HashedIdf(self.idf_full["c2"], default_n), HashedIdf(self.idf_full["c4"], default_a),
                    default_n, default_a)
        # reverse channels: each pool record -> its top S1 records (competition from the S1 side)
        self.rev = {}
        for ch, src in (("c6", "c5"), ("c9", "c2")):
            k = self.K.get(ch, 0)
            if k > 0:
                r, c, s, rk = sparse_topk(self.P[src], self.S[src], k)   # rows: pool, cols: s1
                o = np.argsort(c, kind="stable")
                self.rev[ch] = (c[o], r[o], s[o], rk[o])                # sorted by s1 row
        # c8: pool records WITHOUT an address can only be found by name, where common words crowd them out of
        # the 5 reverse slots; they get a deeper reverse name search
        k8 = self.K.get("c8", 0)
        rows8 = np.flatnonzero(pool["a_core"].astype(object).fillna("").astype(str).str.len().to_numpy() == 0)
        if k8 > 0 and len(rows8):
            r, c, s, rk = sparse_topk(self.P["c2"][rows8], self.S["c2"], k8)
            o = np.argsort(c, kind="stable")
            self.rev["c8"] = (c[o], rows8[r][o], s[o], rk[o])
        self.postal1 = s1["a_postal"].astype(object).fillna("").to_numpy(object)
        self.postalp = pool["a_postal"].astype(object).fillna("").to_numpy(object)
        log.info("  index: S1=%d pool=%d nnz/pool-row name %.1f addr %.1f joint %.1f  (%.0fs)", len(s1), len(pool),
                 self.P["c2"].nnz / max(len(pool), 1), self.P["c4"].nnz / max(len(pool), 1),
                 self.P["c5"].nnz / max(len(pool), 1), time.time() - t0)

    def query(self, rows):
        """Union of all channels for the S1 records at positions `rows` (partition-local)."""
        rows = np.asarray(rows, np.int64)
        K = self.K
        parts = []
        for ch in ("c2", "c4", "c5", "c7", "c1"):
            if K.get(ch, 0) > 0:
                r, c, s, rk = sparse_topk(self.S[ch][rows], self.P[ch], K[ch], BT=self.PT[ch])
                parts.append((ch, rows[r], c, s, rk))
        lo, hi = rows.min(), rows.max()
        member = np.zeros(hi - lo + 1, bool)
        member[rows - lo] = True
        for ch, (s1r, pr, s, rk) in self.rev.items():
            a, b = np.searchsorted(s1r, lo), np.searchsorted(s1r, hi, side="right")
            m = member[s1r[a:b] - lo]
            parts.append((ch, s1r[a:b][m], pr[a:b][m], s[a:b][m], rk[a:b][m]))
        n2 = self.n_pool
        if not parts or not sum(len(p[1]) for p in parts):
            return None
        keys = np.unique(np.concatenate([p[1] * n2 + p[2] for p in parts]))
        pairs = pd.DataFrame({"s1": keys // n2, "cand": keys % n2})
        n_ch = np.zeros(len(keys), np.int8)
        cols = {}
        for ch in CHANNELS:
            cols[f"s_{ch}"] = np.zeros(len(keys), np.float32)
            cols[f"k_{ch}"] = np.full(len(keys), K.get(ch, 0) + 1, np.float32)
        for ch, r, c, s, rk in parts:
            pos = np.searchsorted(keys, r * n2 + c)
            np.maximum.at(cols[f"s_{ch}"], pos, s)
            cols[f"k_{ch}"][pos] = np.minimum(cols[f"k_{ch}"][pos], rk)
            hit = np.zeros(len(keys), bool)
            hit[pos] = True
            n_ch += hit
        for k, v in cols.items():
            pairs[k] = v
        pairs["n_ch"] = n_ch
        ia, ib = pairs["s1"].to_numpy(), pairs["cand"].to_numpy()
        src = {"n_cos_c": "c2", "n_cos_w": "c2", "n_cos_s": None, "a_cos_c": "c4", "a_cos_w": "c4",
               "j_cos_c": "c5"}
        for _, col in EXACT:
            ch = src[col]
            pairs[col] = rowwise_cos(self.S[ch], self.P[ch], ia, ib) if ch else np.float32(0.0)
        pa, pb = self.postal1[ia], self.postalp[ib]
        eq = np.zeros(len(pairs), np.float32)
        conf = np.zeros(len(pairs), np.float32)
        for i, (a, b) in enumerate(zip(pa, pb)):
            if a and b:
                if set(a.split()) & set(b.split()):
                    eq[i] = 1.0
                else:
                    conf[i] = 1.0
        pairs["post_eq"], pairs["post_conf"] = eq, conf
        return pairs


__all__ = ["PartitionIndex", "HashedIdf", "hash_transform", "group_rank", "_empty"]
