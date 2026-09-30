"""Progress tracking for long runs (Kaggle): numbered stages, elapsed time, memory, live progress.json.

Every stage start/end is logged (stdout is flushed per record, so `!python -m src.run` streams in a
notebook) and the partial report is rewritten atomically to reports/progress.json after each stage,
so a crash or a session timeout still leaves every metric computed so far on disk.
"""
from __future__ import annotations

import json
import logging
import os
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np

log = logging.getLogger("ber")


def rss_gb():
    try:
        with open("/proc/self/status", encoding="ascii") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 2**20, 2)
    except Exception:
        pass
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 2**30, 2)
    except Exception:
        return None


def gpu_mem_gb():
    try:
        import torch
        if torch.cuda.is_available():
            return round(max(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 2**30, 2)
    except Exception:
        pass
    return None


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def dump_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in obj.items() if not str(k).startswith("_")} if isinstance(obj, dict) else obj,
                  f, indent=2, default=_json_default)
    os.replace(tmp, path)


def fmt_dur(sec):
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m{sec % 60:02d}s" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


class Tracker:
    def __init__(self, report: dict, path=None, prefix=""):
        self.report = report
        self.path = path
        self.prefix = prefix
        self.t0 = time.time()
        self.i = 0

    def flush(self):
        if self.path:
            try:
                self.report.setdefault("progress", {})["elapsed_sec"] = round(time.time() - self.t0, 1)
                dump_json(self.report, self.path)
            except Exception as e:  # tracking must never break the run
                log.warning("could not write %s: %r", self.path, e)

    @contextmanager
    def stage(self, name):
        self.i += 1
        label = f"{self.prefix}{name}"
        t = time.time()
        log.info(">> [stage %d] %s | elapsed %s | RSS %s GB", self.i, label, fmt_dur(t - self.t0), rss_gb())
        prog = self.report.setdefault("progress", {})
        prog["current_stage"] = label
        prog.setdefault("completed", [])
        self.flush()
        try:
            yield
        except Exception as e:
            prog["failed_stage"] = label
            prog["error"] = repr(e)[:2000]
            prog["traceback"] = traceback.format_exc()[-6000:]
            self.flush()
            log.error("!! stage %s failed after %s: %r", label, fmt_dur(time.time() - t), e)
            raise
        dt = time.time() - t
        self.report.setdefault("timings_sec", {})[label] = round(dt, 1)
        mem = {"rss_gb": rss_gb()}
        g = gpu_mem_gb()
        if g is not None:
            mem["gpu_peak_gb"] = g
        self.report.setdefault("memory_gb", {})[label] = mem
        prog["completed"].append(label)
        prog["current_stage"] = None
        self.flush()
        log.info("<< [stage %d] %s done in %s | total %s | RSS %s GB%s", self.i, label, fmt_dur(dt),
                 fmt_dur(time.time() - self.t0), mem["rss_gb"],
                 f" | GPU peak {g} GB" if g is not None else "")
        if os.environ.get("BER_FAIL_AFTER") == label:           # fault injection for the resume self-test
            raise RuntimeError(f"injected failure after stage {label} (BER_FAIL_AFTER)")


def write_table(df, path_stem):
    """Parquet when pyarrow/fastparquet is available, else gzipped CSV. Returns the written path."""
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    try:
        p = path_stem.with_suffix(".parquet")
        df.to_parquet(p, index=False)
        return p
    except Exception:
        p = path_stem.with_suffix(".csv.gz")
        df.to_csv(p, index=False, compression="gzip")
        return p
