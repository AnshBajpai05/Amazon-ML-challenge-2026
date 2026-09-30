"""End-to-end self-test: run this first on a new machine / Kaggle session (a few minutes).

    python -m src.selftest [--entities 400] [--loco] [--keep DIR] [--set key=value ...]

1. environment: imports, versions, rapidfuzz cpdist, sparse_dot_topn, GBDT backend and torch GPU probes
2. synthetic dataset with the challenge schema (US + India train, France-only-in-test)
3. run #1 with an injected crash after pass-1          -> must fail and leave progress.json
4. run #2 (resume)                                     -> must hit the checkpoints written by run #1
5. run #3 (fully cached)                               -> outputs byte-identical to run #2 (determinism)
6. score vs synthetic gold (all countries incl. France), validator PASS, zip layout, report keys
Exit code 0 and 'SELFTEST PASS' when every check holds.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

log = logging.getLogger("ber")


def env_check():
    rows, ok = [], True
    required = ["numpy", "scipy", "pandas", "sklearn", "lightgbm", "rapidfuzz", "joblib", "yaml"]
    optional = ["xgboost", "sparse_dot_topn", "torch", "transformers", "pyarrow", "psutil"]
    for m in required + optional:
        try:
            v = getattr(importlib.import_module(m), "__version__", "?")
            rows.append((m, v, "ok"))
        except Exception as e:
            rows.append((m, None, f"MISSING ({type(e).__name__})"))
            if m in required:
                ok = False
    from rapidfuzz import process
    rows.append(("rapidfuzz.cpdist", "yes" if hasattr(process, "cpdist") else "no (slow fallback)", "ok"))
    from .compat import effective_cpus
    rows.append(("CPUs", f"{effective_cpus()} usable", f"os.cpu_count()={os.cpu_count()}"))
    from .models import probe_xgb_cuda
    g_ok, g_why = probe_xgb_cuda()
    rows.append(("xgboost CUDA", "usable" if g_ok else "no", g_why[:80]))
    try:
        from .neural import torch_cuda_status
        t = torch_cuda_status()
        rows.append(("torch CUDA", "usable" if t.get("ok") else "no", str(t)[:120]))
    except Exception as e:
        rows.append(("torch CUDA", "no", repr(e)[:80]))
    for r in rows:
        print(f"  {r[0]:<18} {str(r[1]):<24} {r[2]}")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", type=int, default=400)
    ap.add_argument("--keep", default=None, help="work dir to keep (default: temp dir, removed on success)")
    ap.add_argument("--loco", action="store_true", help="also run leave-one-country-out on the synthetic data")
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args(argv)
    t0 = time.time()
    fails = []

    def check(cond, msg):
        print(("  [ok]   " if cond else "  [FAIL] ") + msg)
        if not cond:
            fails.append(msg)

    print("== 1. environment")
    check(env_check(), "required packages importable")

    work = Path(a.keep) if a.keep else Path(tempfile.mkdtemp(prefix="ber_selftest_"))
    if work.exists() and a.keep:
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    print(f"== 2. synthetic data in {work}")
    from .synth import main as synth_main
    synth_main(["--out", str(work / "dataset"), "--entities", str(a.entities), "--seed", "7"])

    from .run import main as run_main
    base = ["--data", str(work), "--cache", str(work / "ber_cache"), "--team", "selftest",
            "--set", "neural.ce_max_train=2000", "--set", "seeds_gpu=[42]", *sum((["--set", s] for s in a.set), [])]

    def run(tag, extra=(), fail_after=None):
        if fail_after:
            os.environ["BER_FAIL_AFTER"] = fail_after
        try:
            return run_main([*base, "--out", str(work / tag / "output"), "--report", str(work / tag / "reports"),
                             *extra])
        finally:
            os.environ.pop("BER_FAIL_AFTER", None)

    print("== 3. run #1 with an injected crash after pass1")
    crashed = False
    try:
        run("run1", fail_after="pass1")
    except RuntimeError as e:
        crashed = "injected failure" in str(e)
    check(crashed, "run #1 stopped by the injected failure")
    prog = json.loads((work / "run1" / "reports" / "progress.json").read_text(encoding="utf-8"))
    check("pass1" in prog.get("progress", {}).get("completed", []), "progress.json recorded the completed stages")

    print("== 4. run #2 resumes from checkpoints")
    r2 = run("run2", ["--zip", str(work / "run2")])
    hits = r2.get("checkpoints", {}).get("hits", [])
    check(any(h.startswith("normalize-") for h in hits) and any(h.startswith("pass1-") for h in hits),
          f"run #2 reused checkpoints from run #1 ({len(hits)} hits)")

    print("== 5. run #3 fully cached, byte-identical outputs")
    r3 = run("run3")
    check(r3["output_sha256"] == r2["output_sha256"], "outputs identical across runs (deterministic)")
    check(any(h.startswith("pass2-") for h in r3["checkpoints"]["hits"]), "run #3 loaded pass-2 from checkpoint")

    print("== 6. quality and packaging")
    from .evaluate import score_file
    out2 = work / "run2" / "output"
    sc = score_file(out2 / "matching_results.tsv", work / "dataset" / "_synthetic_test_ground_truth.tsv",
                    out2 / "candidate_pairs.tsv", work / "dataset" / "test")
    by_c = {k: round(v["f05"], 4) for k, v in sc.get("by_country", {}).items()}
    print(f"  synthetic test macro F0.5 = {sc['macro_f05']:.4f}  by country {by_c}  PC {sc['blocking']['PC']:.4f}")
    check(sc["macro_f05"] > 0.85, "synthetic test F0.5 > 0.85")
    check(by_c.get("france", 0) > 0.8, "unseen country (France) handled: F0.5 > 0.8")
    check(r2["local_check"] == ["PASS"], "local validator PASS")
    check(r2["oof"]["nested_f05"] > 0.85, f"OOF nested F0.5 = {r2['oof']['nested_f05']:.4f} > 0.85")
    zp = Path(r2["submission_zip"])
    names = zipfile.ZipFile(zp).namelist()
    need = ["output/matching_results.tsv", "output/candidate_pairs.tsv", "Documentation_template.md",
            "code/business_entity_resolution/README.md", "code/business_entity_resolution/requirements.txt",
            "code/business_entity_resolution/src/run.py"]
    check(all(n in names for n in need), f"zip layout ({len(names)} files)")
    rep_dir = work / "run2" / "reports"
    for f in ("run_report.json", "progress.json", "run.log", "Documentation_template.md", "oof_errors.tsv",
              "feature_importance.csv", "decoder_grid.csv"):
        check((rep_dir / f).exists(), f"report artifact {f}")
    check(any((rep_dir / f"oof_pairs{s}").exists() for s in (".parquet", ".csv.gz")), "report artifact oof_pairs")
    for k in ("eda", "blocking", "models", "calibration", "decoder", "oof", "test_dashboard", "timings_sec",
              "memory_gb", "output_sha256"):
        check(k in r2, f"run_report has '{k}'")

    print("== 6b. large-data path (mode=scale: partitioned, chunked, word-level blocking)")
    r4 = run("run4", ["--set", "mode=scale", "--set", "scale.train_s1=100000", "--set", "scale.chunk_s1=300"])
    out4 = work / "run4" / "output"
    sc4 = score_file(out4 / "matching_results.tsv", work / "dataset" / "_synthetic_test_ground_truth.tsv",
                     out4 / "candidate_pairs.tsv", work / "dataset" / "test")
    print(f"  scale path: synthetic test macro F0.5 = {sc4['macro_f05']:.4f}  "
          f"by country { {k: round(v['f05'], 4) for k, v in sc4.get('by_country', {}).items()} }")
    check(r4.get("mode") == "scale", "run #4 used the scale path")
    check(r4["local_check"] == ["PASS"], "scale path: local validator PASS")
    check(sc4["macro_f05"] > 0.8, "scale path: synthetic test F0.5 > 0.8 (word-level blocking only)")

    if a.loco:
        print("== 7. leave-one-country-out")
        from .evaluate import main as eval_main
        res = eval_main(["--data", str(work), "--report", str(work / "loco"), "--cache", str(work / "ber_cache"),
                         *sum((["--set", s] for s in a.set), [])])
        check(len(res) >= 2, f"LOCO produced {len(res)} held-out scores: "
                             f"{ {k: round(v['heldout']['macro_f05'], 4) for k, v in res.items()} }")

    print(f"== selftest finished in {time.time() - t0:.0f}s")
    if fails:
        print("SELFTEST FAIL:", *fails, sep="\n  - ")
        print(f"work dir kept for inspection: {work}")
        sys.exit(1)
    print("SELFTEST PASS")
    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    main()
