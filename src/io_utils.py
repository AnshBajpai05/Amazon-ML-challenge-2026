"""Safe TSV IO, dataset discovery, writers and local output checks (plan A.1)."""
from __future__ import annotations

import csv
import hashlib
import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

log = logging.getLogger("ber")

SRC_COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path) -> pd.DataFrame:
    """Never let pandas guess: 'NA'/'None' can be real business names, and quotes are data."""
    try:
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[],
                         quoting=csv.QUOTE_NONE, encoding="utf-8-sig")
    except (pd.errors.ParserError, UnicodeDecodeError) as e:
        log.warning("pandas could not parse %s (%s); using the tolerant reader", path, e)
        df = _read_tsv_tolerant(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def _read_tsv_tolerant(path) -> pd.DataFrame:
    """Line-by-line reader: surplus tabs in a source row are folded into the address column."""
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
        lines = f.read().split("\n")
    header = lines[0].rstrip("\r").split("\t")
    n, rows = len(header), []
    for ln in lines[1:]:
        ln = ln.rstrip("\r")
        if not ln.strip():
            continue
        parts = ln.split("\t")
        if len(parts) > n and n >= 4:
            parts = parts[: n - 2] + [" ".join(parts[n - 2: len(parts) - 1]), parts[-1]]
        parts = (parts + [""] * n)[:n]
        rows.append(parts)
    return pd.DataFrame(rows, columns=header, dtype=str)


def read_source(path) -> pd.DataFrame:
    df = read_tsv(path)
    for c in SRC_COLS:
        if c not in df.columns:
            log.warning("%s: column %r missing, filled with empty strings", path, c)
            df[c] = ""
    df = df[SRC_COLS].copy()
    df["entity_id"] = df["entity_id"].str.strip()
    df = df[df["entity_id"] != ""]
    dup = df["entity_id"].duplicated()
    if dup.any():
        log.warning("%s: %d duplicate entity_id rows dropped", path, int(dup.sum()))
        df = df[~dup]
    return df.reset_index(drop=True)


def find_data_dir(root) -> Path:
    """Accepts the dataset dir itself or any ancestor (e.g. /kaggle/input)."""
    root = Path(root)
    if (root / "train" / "train_source1.tsv").exists():
        return root
    hits = sorted(root.rglob("train_source1.tsv"))
    for h in hits:
        d = h.parent.parent
        if (d / "test" / "test_source1.tsv").exists():
            return d
    if hits:
        return hits[0].parent.parent
    raise FileNotFoundError(f"no train/train_source1.tsv under {root}")


def find_validator(data_dir, extra_roots=("/kaggle/input",)) -> Path | None:
    d = Path(data_dir).resolve()
    for p in [d, *d.parents][:4]:
        c = p / "utils" / "validate_submission.py"
        if c.exists():
            return c
    for r in extra_roots:
        if os.path.isdir(r):
            for c in sorted(Path(r).rglob("validate_submission.py")):
                return c
    return None


def load_gold(path) -> dict:
    g = read_tsv(path)
    out = {}
    for s, m in zip(g["source1_entity_id"].str.strip(), g["matched_entity_ids"]):
        out[s] = [x.strip() for x in m.split(",") if x.strip()]
    return out


def load_split(data_dir, split: str) -> dict:
    """Returns s1 (file order kept for the writer), pool = S2 u S3 sorted by id, and gold (train only)."""
    d = Path(data_dir) / split
    s1 = read_source(d / f"{split}_source1.tsv")
    s2 = read_source(d / f"{split}_source2.tsv")
    s3 = read_source(d / f"{split}_source3.tsv")
    pool = pd.concat([s2, s3], ignore_index=True)
    pool = pool[~pool["entity_id"].duplicated()].sort_values("entity_id", kind="stable").reset_index(drop=True)
    s1_order = s1["entity_id"].tolist()
    s1 = s1.sort_values("entity_id", kind="stable").reset_index(drop=True)
    gold = None
    gp = d / f"{split}_ground_truth.tsv"
    if gp.exists():
        raw = load_gold(gp)
        pool_ids = set(pool["entity_id"])
        gold, missing, dropped = {}, 0, 0
        for s in s1["entity_id"]:
            if s not in raw:
                missing += 1
            ids = list(dict.fromkeys(raw.get(s, [])))
            ok = [x for x in ids if x in pool_ids]
            dropped += len(ids) - len(ok)
            gold[s] = ok
        if missing or dropped:
            log.warning("%s gold: %d S1 rows missing (treated as singletons), %d ids not in pool dropped",
                        split, missing, dropped)
    log.info("%s: S1=%d  S2=%d  S3=%d  pool=%d", split, len(s1), len(s2), len(s3), len(pool))
    return {"s1": s1, "pool": pool, "gold": gold, "s1_order": s1_order}


def write_ids(path, s1_ids, id_map, col):
    """col = 'matched_entity_ids' or 'candidate_entity_ids'. Driven by the S1 id list,
    so every test entity (France included) gets exactly one row; empty list = empty field."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for s in s1_ids:
            ids = list(dict.fromkeys(id_map.get(s, [])))
            f.write(f"{s}\t{','.join(ids)}\n")


def check_outputs(match_path, cand_path, s1_ids, pool_ids) -> list:
    """Local mirror of the published rules. Still run utils/validate_submission.py before uploading."""
    errs, lists = [], {}
    pool_ids, s1_set = set(pool_ids), set(s1_ids)
    for path, col in ((match_path, "matched_entity_ids"), (cand_path, "candidate_entity_ids")):
        d = read_tsv(path)
        if list(d.columns) != ["source1_entity_id", col]:
            errs.append(f"{path}: header {list(d.columns)}")
            continue
        if d["source1_entity_id"].duplicated().any():
            errs.append(f"{path}: duplicate source1_entity_id rows")
        if set(d["source1_entity_id"]) != s1_set:
            errs.append(f"{path}: S1 ids differ from test_source1")
        lists[col] = {}
        for s, lst in zip(d["source1_entity_id"], d[col]):
            ids = [x for x in lst.split(",") if x]
            lists[col][s] = set(ids)
            if len(ids) != len(set(ids)):
                errs.append(f"{path}: duplicate ids for {s}")
            if any(x[:3] not in ("S2-", "S3-") or x not in pool_ids for x in ids):
                errs.append(f"{path}: id outside the test S2/S3 pool for {s}")
    for s, m in lists.get("matched_entity_ids", {}).items():
        if not m <= lists.get("candidate_entity_ids", {}).get(s, set()):
            errs.append(f"match not in candidates for {s}")
    return errs[:50] or ["PASS"]


def run_official_validator(validator, match_path, cand_path, test_dir):
    cmd = [sys.executable, str(validator), "--matching", str(match_path),
           "--candidate", str(cand_path), "--test-dir", str(test_dir)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        return r.returncode == 0, (r.stdout + r.stderr).strip()
    except Exception as e:  # validator missing or crashed: report, never hide
        return False, repr(e)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
