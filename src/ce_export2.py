"""Phase-2 cross-encoder kit: the uncertain pairs of the final (v4) holdout and test that phase 1 did not score.

Phase 1 (src/ce_export.py) exported the uncertain pairs of the first candidate set; the final blocking recovered new
candidates (native-script, website, address-less records) that have no transformer score yet. This writes a
score-only kit: the models trained in phase 1 are reused (attached on Kaggle as the previous notebook's output).

    python -m src.ce_export2 --out ../../kaggle_ce2
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .proxy import proxy_key, region_folds
from .scale import normalized_source


def _covered(ce_zip, split):
    with zipfile.ZipFile(ce_zip) as z:
        name = next(n for n in z.namelist() if n.endswith(f"scores_{split}.parquet"))
        d = pd.read_parquet(z.open(name), columns=["s1_id", "cand_id"])
    return pd.MultiIndex.from_arrays([d["s1_id"].astype(object), d["cand_id"].astype(object)])


def _new_pairs(s1_ids, cand_ids, covered):
    key = pd.MultiIndex.from_arrays([pd.Index(s1_ids, dtype=object), pd.Index(cand_ids, dtype=object)])
    return ~key.isin(covered)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[3]
    ap.add_argument("--out", default=str(root / "kaggle_ce2"))
    ap.add_argument("--cache", default=str(root / "ber_cache"))
    ap.add_argument("--data", default=str(root / "student_resource" / "dataset"))
    ap.add_argument("--report", default=str(Path(__file__).resolve().parents[1] / "reports_v4"))
    ap.add_argument("--ce", default=str(root / "ce_outputs.zip"))
    ap.add_argument("--kit1", default=str(root / "kaggle_ce"))
    ap.add_argument("--band", type=float, default=0.002)
    a = ap.parse_args(argv)
    t0 = time.time()
    out, data = Path(a.out), Path(a.out) / "data"
    data.mkdir(parents=True, exist_ok=True)
    lo, hi = a.band, 1 - a.band
    man = json.loads((Path(a.report) / "bigtrain" / "manifest.json").read_text(encoding="utf-8"))
    d = Path(a.data)
    paths = {sp: {k: normalized_source(d / sp / f"{sp}_source{k}.tsv", Path(a.cache)) for k in (1, 2, 3)}
             for sp in ("train", "test")}

    # ---- holdout (final 60k/country holdout scored with the final models)
    P = joblib.load(Path(a.cache) / f"{proxy_key(60000, tag=man['pass2'])}.joblib")
    from .decide import proxy_pair_ids
    s1_ids, cand_ids = proxy_pair_ids(P, a.data, a.cache)
    p = P["p2"].astype(np.float64)
    m = (p >= lo) & (p <= hi)
    m &= _new_pairs(s1_ids, cand_ids, _covered(a.ce, "holdout"))
    folds = region_folds(P)
    ho = pd.DataFrame({"s1_id": s1_ids[m], "cand_id": cand_ids[m], "cty": P["cty"][P["s"][m]],
                       "y": P["y"][m].astype(np.int8), "p_gbdt": P["p2"][m].astype(np.float32),
                       "fold": folds[P["s"][m]].astype(np.int8)})
    ho.to_parquet(data / "holdout_pairs.parquet", index=False)
    print(f"holdout: {len(ho):,} new uncertain pairs ({ho.y.mean():.3f} true) {time.time() - t0:.0f}s", flush=True)
    del P, s1_ids, cand_ids

    # ---- test (final test probabilities)
    keys = json.loads((Path(a.report) / "bigtrain" / "test_keys.json").read_text(encoding="utf-8"))
    cov_t = _covered(a.ce, "test")
    parts = []
    for part in sorted(keys):
        s, c, p2, meta, p1 = joblib.load(Path(a.cache) / f"{keys[part]}.joblib")
        i1 = pq.read_table(paths["test"][1], columns=["id"], filters=[("cty", "==", part)]).column(0).to_numpy(
            zero_copy_only=False)
        ip = np.concatenate([pq.read_table(paths["test"][k], columns=["id"], filters=[("cty", "==", part)])
                             .column(0).to_numpy(zero_copy_only=False) for k in (2, 3)])
        m = (p2 >= lo) & (p2 <= hi)
        sid, cid = i1[s[m]], ip[c[m]]
        new = _new_pairs(sid, cid, cov_t)
        parts.append(pd.DataFrame({"s1_id": sid[new], "cand_id": cid[new], "cty": part,
                                   "p_gbdt": p2[m][new].astype(np.float32)}))
        print(f"  test {part}: {int(m.sum()):,} uncertain, {int(new.sum()):,} new", flush=True)
        del s, c, p2, meta, p1, i1, ip
    te = pd.concat(parts, ignore_index=True)
    te.to_parquet(data / "test_pairs.parquet", index=False)
    print(f"test: {len(te):,} new uncertain pairs {time.time() - t0:.0f}s", flush=True)

    # ---- train pairs of phase 1 (only its validation rows are scored; the file is required by run.py)
    shutil.copy(Path(a.kit1) / "data" / "train_pairs.parquet", data / "train_pairs.parquet")
    trv = pd.read_parquet(data / "train_pairs.parquet")
    trv = trv[trv.is_val]

    # ---- texts
    need = {"train": set(ho.s1_id) | set(ho.cand_id) | set(trv.s1_id) | set(trv.cand_id),
            "test": set(te.s1_id) | set(te.cand_id)}
    recs = []
    for split in ("train", "test"):
        vs = pa.array(sorted(need[split]))
        for k in (1, 2, 3):
            tb = pq.read_table(paths[split][k], columns=["id", "raw_name", "raw_addr"])
            tb = tb.filter(pc.is_in(tb.column("id"), value_set=vs))
            recs.append(tb.rename_columns(["id", "name", "addr"]))
    rec = pa.concat_tables(recs)
    pq.write_table(rec, data / "records.parquet", compression="zstd")
    manifest = {"phase": 2, "holdout_pairs": len(ho), "test_pairs": len(te), "records": rec.num_rows,
                "band": [lo, hi], "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (data / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    shutil.copy(Path(a.kit1) / "run.py", out / "run.py")
    print(f"done {json.dumps(manifest)} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
