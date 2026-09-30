"""Small compatibility shims so the pipeline runs across library versions and container setups."""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from rapidfuzz import process


@lru_cache(maxsize=1)
def effective_cpus() -> int:
    """CPUs this process may actually use: min(cpu_count, affinity, cgroup v1/v2 quota).
    Containers (Kaggle, Docker) often report the host's CPU count while the quota is far lower; sizing
    OpenMP/joblib pools by os.cpu_count() then oversubscribes and LightGBM slows down by orders of magnitude.
    Override with BER_THREADS."""
    if os.environ.get("BER_THREADS", "").isdigit():
        return max(1, int(os.environ["BER_THREADS"]))
    n = os.cpu_count() or 1
    try:
        n = min(n, len(os.sched_getaffinity(0)))
    except Exception:
        pass
    try:                                                    # cgroup v2: "<quota> <period>" or "max <period>"
        with open("/sys/fs/cgroup/cpu.max", encoding="ascii") as f:
            q, p = f.read().split()[:2]
        if q != "max":
            n = min(n, max(1, int(float(q) / float(p))))
    except Exception:
        pass
    try:                                                    # cgroup v1
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", encoding="ascii") as f:
            q = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", encoding="ascii") as f:
            p = int(f.read())
        if q > 0 and p > 0:
            n = min(n, max(1, q // p))
    except Exception:
        pass
    return max(1, n)


def cpdist(a, b, scorer, workers=None):
    """Element-wise scorer(a[i], b[i]) as float32. rapidfuzz >= 3.6 has a parallel C++ cpdist;
    older versions fall back to a loop over the (still C-implemented) scorer."""
    workers = effective_cpus() if workers in (None, -1) else workers
    if hasattr(process, "cpdist"):
        try:
            return np.asarray(process.cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32), np.float32)
        except TypeError:  # very early cpdist signatures without dtype
            return np.asarray(process.cpdist(a, b, scorer=scorer, workers=workers), np.float32)
    return np.fromiter((scorer(x, y) for x, y in zip(a, b)), dtype=np.float32, count=len(a))
