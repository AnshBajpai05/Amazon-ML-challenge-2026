"""Export the pairs a cross-encoder should see (Kaggle GPU, kaggle_ce/run.py) from the local caches.

  train_pairs.parquet    training S1s only (the original 200k sample): hard pairs (GBDT OOF p in the band), a share
                         of the confident ones; is_val = 5% of S1s (grouped) for monitoring. Never holdout S1s.
  holdout_pairs.parquet  holdout (src/proxy.py) pairs in the band, with labels and city folds - for measuring and
                         fitting the fusion honestly.
  test_pairs.parquet     test pairs in the band (where a decision can change).
  records.parquet        id, name, addr (raw strings) of every record referenced above.
  manifest.json          counts and settings.

    python -m src.ce_export --out ../../kaggle_ce/data [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .cache import Checkpointer
from .proxy import proxy_key, region_folds
from .scale import normalized_source


def _pool_ids(paths, split, part):
    """Pool ids of a partition in load_partition order (source2 then source3, file order)."""
    return np.concatenate([pq.read_table(paths[split][k], columns=["id"], filters=[("cty", "==", part)])
                           .column(0).to_numpy(zero_copy_only=False) for k in (2, 3)])


def _s1_val(ids):
    return np.array([int(hashlib.md5(s.encode()).hexdigest()[:8], 16) % 20 == 0 for s in ids])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[3]
    ap.add_argument("--out", default=str(root / "kaggle_ce" / "data"))
    ap.add_argument("--cache", default=str(root / "ber_cache"))
    ap.add_argument("--data", default=str(root / "student_resource" / "dataset"))
    ap.add_argument("--report", default=str(Path(__file__).resolve().parents[1] / "reports"))
    ap.add_argument("--n", type=int, default=150000, help="holdout size per country (proxy key)")
    ap.add_argument("--band", type=float, default=0.002, help="uncertain band [band, 1-band] of GBDT p")
    ap.add_argument("--keep-pos", dest="keep_pos", type=float, default=0.3, help="share of confident positives kept")
    ap.add_argument("--keep-neg", dest="keep_neg", type=float, default=0.04, help="share of confident negatives kept")
    ap.add_argument("--smoke", action="store_true", help="tiny export for testing the Kaggle kit")
    a = ap.parse_args(argv)
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = Checkpointer(a.cache)
    d = Path(a.data)
    paths = {sp: {k: normalized_source(d / sp / f"{sp}_source{k}.tsv", Path(a.cache)) for k in (1, 2, 3)}
             for sp in ("train", "test")}
    lo, hi = a.band, 1 - a.band
    rng = np.random.RandomState(0)
    man = {"band": [lo, hi], "created": time.strftime("%Y-%m-%d %H:%M:%S"), "smoke": a.smoke}

    # ---- train pairs: base training S1s with GBDT OOF p (reports/oof_pairs.parquet; column p1 = OOF pass-2 p)
    t = pq.read_table(Path(a.report) / "oof_pairs.parquet", columns=["s1_id", "cand_id", "country", "y", "p1"])
    p = t.column("p1").to_numpy()
    y = t.column("y").to_numpy().astype(bool)
    u = rng.rand(len(p))
    keep = ((p >= lo) & (p <= hi)) | (y & (u < a.keep_pos)) | (~y & (u < a.keep_neg))
    if a.smoke:
        keep &= rng.rand(len(p)) < 0.002
    t = t.filter(pa.array(keep))
    tr = pd.DataFrame({"s1_id": t.column("s1_id").to_numpy(zero_copy_only=False),
                       "cand_id": t.column("cand_id").to_numpy(zero_copy_only=False),
                       "cty": t.column("country").to_numpy(zero_copy_only=False),
                       "y": t.column("y").to_numpy().astype(np.int8),
                       "p_gbdt": t.column("p1").to_numpy().astype(np.float32)})
    del t, p, y, u, keep
    tr["is_val"] = _s1_val(tr["s1_id"].to_numpy(object))
    tr.to_parquet(out / "train_pairs.parquet", index=False)
    man["train_pairs"] = {"n": len(tr), "pos": int(tr.y.sum()), "val": int(tr.is_val.sum()),
                          "s1": int(tr.s1_id.nunique())}
    print(f"train pairs {len(tr):,} (pos {tr.y.mean():.3f}, val {tr.is_val.mean():.3f}) {time.time() - t0:.0f}s",
          flush=True)

    # ---- holdout pairs (proxy built with the submitted models)
    P = ck.load(proxy_key(a.n))
    folds = region_folds(P)
    off = 0
    cand_id = np.empty(len(P["c"]), dtype=object)
    for part in ["india", "us"]:                              # proxy partition order (sorted train & test parts)
        pid = _pool_ids(paths, "train", part)
        m = (P["c"] >= off) & (P["c"] < off + len(pid))
        cand_id[m] = pid[P["c"][m] - off]
        off += len(pid)
    pb = P["p2"].astype(np.float64)
    m = (pb >= lo) & (pb <= hi)
    if a.smoke:
        m &= rng.rand(len(m)) < 0.005
    ho = pd.DataFrame({"s1_id": P["ids"][P["s"][m]], "cand_id": cand_id[m], "cty": P["cty"][P["s"][m]],
                       "y": P["y"][m].astype(np.int8), "p_gbdt": P["p2"][m].astype(np.float32),
                       "fold": folds[P["s"][m]].astype(np.int8)})
    del P, cand_id
    ho.to_parquet(out / "holdout_pairs.parquet", index=False)
    man["holdout_pairs"] = {"n": len(ho), "pos": int(ho.y.sum()), "s1": int(ho.s1_id.nunique())}
    print(f"holdout pairs {len(ho):,} (pos {ho.y.mean():.3f}) {time.time() - t0:.0f}s", flush=True)

    # ---- test pairs (saved probabilities of the submitted models)
    run_rep = json.loads((Path(a.report) / "run_report.json").read_text(encoding="utf-8"))
    from .decide import test_keys
    keys, tpaths = test_keys(run_rep, ck, a.data, a.cache)
    parts = []
    for part in sorted(keys):
        s, c, p2, meta, p1 = ck.load(keys[part])
        m = (p2 >= lo) & (p2 <= hi)
        if a.smoke:
            m &= rng.rand(len(m)) < 0.002
        i1 = pq.read_table(tpaths[1], columns=["id"], filters=[("cty", "==", part)]).column(0).to_numpy(
            zero_copy_only=False)
        ip = _pool_ids(paths, "test", part)
        parts.append(pd.DataFrame({"s1_id": i1[s[m]], "cand_id": ip[c[m]], "cty": part,
                                   "p_gbdt": p2[m].astype(np.float32)}))
        print(f"  test {part}: {int(m.sum()):,} of {len(m):,} pairs in band", flush=True)
        del s, c, p2, meta, p1, i1, ip
    te = pd.concat(parts, ignore_index=True)
    te.to_parquet(out / "test_pairs.parquet", index=False)
    man["test_pairs"] = {"n": len(te), "s1": int(te.s1_id.nunique()),
                         "by_country": te.cty.value_counts().to_dict()}
    print(f"test pairs {len(te):,} {time.time() - t0:.0f}s", flush=True)

    # ---- texts of every referenced record (raw strings: scripts, accents and noise intact)
    need = {"train": set(tr.s1_id) | set(tr.cand_id) | set(ho.s1_id) | set(ho.cand_id),
            "test": set(te.s1_id) | set(te.cand_id)}
    del tr, ho, te
    recs = []
    for split in ("train", "test"):
        vs = pa.array(sorted(need[split]))
        for k in (1, 2, 3):
            tb = pq.read_table(paths[split][k], columns=["id", "raw_name", "raw_addr"])
            tb = tb.filter(pc.is_in(tb.column("id"), value_set=vs))
            recs.append(tb.rename_columns(["id", "name", "addr"]))
            print(f"  records {split} source{k}: {tb.num_rows:,}", flush=True)
            del tb
    rec = pa.concat_tables(recs)
    pq.write_table(rec, out / "records.parquet", compression="zstd")
    man["records"] = rec.num_rows
    (out / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(f"done: {json.dumps(man)} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
