"""Content-addressed stage checkpoints for resumable runs.

Each stage's key hashes everything that determines its output: a fingerprint of the input data,
the config sections it reads, the source code of the modules it runs, and the key of the stage
upstream. Consequences:
  * rerunning after a crash/timeout resumes at the first stage (or fold) that did not finish;
  * changing only the decoder grid reuses everything upstream; changing model parameters reuses
    normalization, blocking and features; editing features.py invalidates features and everything after;
  * a previous Kaggle version's cache can be attached as an input and read via --resume-from.
Writes are atomic (tmp file + rename); a failed write (e.g. disk full) only logs a warning.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger("ber")
SRC = Path(__file__).resolve().parent


def _h(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def code_hash(*modules) -> str:
    h = hashlib.sha1()
    for m in sorted(modules):
        p = SRC / f"{m}.py"
        h.update(m.encode())
        h.update(p.read_bytes() if p.exists() else b"")
    return h.hexdigest()[:16]


def frame_fingerprint(df: pd.DataFrame) -> str:
    if df is None or len(df) == 0:
        return "empty"
    v = pd.util.hash_pandas_object(df.reset_index(drop=True), index=False).to_numpy()
    return hashlib.sha1(np.ascontiguousarray(v).tobytes() + str(list(df.columns)).encode()).hexdigest()[:16]


def split_fingerprint(split: dict) -> str:
    parts = [frame_fingerprint(split["s1"]), frame_fingerprint(split["pool"])]
    if split.get("gold") is not None:
        parts.append(_h(sorted((k, sorted(v)) for k, v in split["gold"].items())))
    return _h(parts)


class Checkpointer:
    def __init__(self, root=None, read_roots=(), enabled=True, min_free_gb=2.0):
        self.root = Path(root) if (root and enabled) else None
        self.read_roots = [Path(r) for r in read_roots if r]
        self.enabled = enabled and self.root is not None
        self.min_free_gb = min_free_gb
        self._warned_disk = False
        self.hits, self.misses = [], []
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def key(self, name, *deps) -> str:
        return f"{name}-{_h(deps)}"

    def _paths(self, key):
        roots = ([self.root] if self.root else []) + self.read_roots
        return [r / f"{key}.joblib" for r in roots]

    def load(self, key):
        if not (self.enabled or self.read_roots):
            return None
        for p in self._paths(key):
            if p.exists():
                try:
                    t = time.time()
                    obj = joblib.load(p)
                    self.hits.append(key)
                    log.info("   checkpoint hit: %s (%.1fs, %s)", key, time.time() - t, p.parent)
                    return obj
                except Exception as e:
                    log.warning("   unreadable checkpoint %s (%r): recomputing", p, e)
        self.misses.append(key)
        return None

    def save(self, key, obj):
        if not self.enabled:
            return
        p = self.root / f"{key}.joblib"
        tmp = p.with_suffix(".tmp")
        try:
            free = shutil.disk_usage(self.root).free
            if free < self.min_free_gb * 2**30:              # Kaggle /kaggle/working is capped (~20 GB)
                if not self._warned_disk:
                    log.warning("   checkpoints paused: only %.1f GB free (< %.1f GB)", free / 2**30, self.min_free_gb)
                    self._warned_disk = True
                return
            t = time.time()
            joblib.dump(obj, tmp, compress=1)                 # zlib-1: fast, typically 2-4x smaller
            os.replace(tmp, p)
            log.info("   checkpoint saved: %s (%.1f MB, %.1fs)", key, p.stat().st_size / 2**20, time.time() - t)
        except Exception as e:  # disk full etc.: the run goes on without this checkpoint
            log.warning("   could not save checkpoint %s: %r", key, e)
            try:
                tmp.unlink()
            except Exception:
                pass

    def cached(self, key, fn):
        obj = self.load(key)
        if obj is None:
            obj = fn()
            self.save(key, obj)
        return obj

    def summary(self):
        size = sum(p.stat().st_size for p in self.root.glob("*.joblib")) if self.enabled else 0
        return {"enabled": self.enabled, "root": str(self.root) if self.root else None,
                "read_roots": [str(r) for r in self.read_roots], "hits": self.hits, "misses": self.misses,
                "size_mb": round(size / 2**20, 1)}


NULL = Checkpointer(enabled=False)
