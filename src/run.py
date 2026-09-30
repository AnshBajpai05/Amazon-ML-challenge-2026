"""Single entry point: data -> blocking -> matching -> output (plan section 13).

    python -m src.run --data /path/to/student_resource/dataset --out ../../output
    python -m src.run --data /kaggle/input --out /kaggle/working/output --team myteam --zip /kaggle/working

--data may be the dataset folder or any ancestor of it (it is searched for train/train_source1.tsv).
"""
from __future__ import annotations

import argparse
import logging
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from .cache import Checkpointer
from .compat import effective_cpus
from .eda import gates
from .io_utils import (check_outputs, find_data_dir, find_validator, load_split, run_official_validator, sha256,
                       write_ids)
from .track import Tracker, dump_json, fmt_dur  # noqa: F401  (dump_json re-exported for evaluate.py)

log = logging.getLogger("ber")
CODE_DIR = Path(__file__).resolve().parent.parent          # code/business_entity_resolution


def load_config(path, overrides=()):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for ov in overrides:                                    # --set a.b.c=value (value parsed as YAML)
        key, val = ov.split("=", 1)
        d = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = yaml.safe_load(val)
    nc = cfg.setdefault("neural", {})
    for k in ("cross_encoder", "dense"):                    # YAML reads bare on/off as booleans
        if isinstance(nc.get(k), bool):
            nc[k] = "on" if nc[k] else "off"
    return cfg


def setup_logging(report_dir):
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    root = logging.getLogger("ber")
    root.setLevel(logging.INFO)
    root.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(Path(report_dir) / "run.log", "w", "utf-8")):
        h.setFormatter(fmt)
        root.addHandler(h)


def env_info():
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "effective_cpus": effective_cpus()}
    for m in ("numpy", "scipy", "pandas", "sklearn", "lightgbm", "xgboost", "rapidfuzz", "sparse_dot_topn",
              "joblib", "yaml", "torch", "transformers"):
        try:
            info[m] = __import__(m).__version__
        except Exception:
            info[m] = None
    try:
        info["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                                      "--format=csv,noheader"], capture_output=True, text=True,
                                     timeout=20).stdout.strip()
    except Exception:
        info["gpu"] = None
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        pass
    return info


def peak_rss_gb():
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)   # KB on Linux
    except Exception:
        try:
            import psutil
            return round(psutil.Process().memory_info().peak_wset / 2**30, 2)
        except Exception:
            return None


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("BER_DATA", "/kaggle/input"))
    ap.add_argument("--out", default=str(CODE_DIR.parent.parent / "output"))
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--report", default=str(CODE_DIR / "reports"))
    ap.add_argument("--set", action="append", default=[], help="config override, e.g. --set backend=lightgbm")
    ap.add_argument("--team", default="team")
    ap.add_argument("--zip", default=None, help="directory to write <team>_submission.zip into")
    ap.add_argument("--cache", default=None, help="checkpoint dir (default: <out>/../ber_cache)")
    ap.add_argument("--resume-from", action="append", default=[],
                    help="extra read-only checkpoint dirs, e.g. a previous Kaggle version's ber_cache")
    ap.add_argument("--no-cache", action="store_true", help="disable checkpoints (always recompute)")
    ap.add_argument("--budget-hours", type=float, default=None, help="time budget; optional stages are skipped "
                                                                      "when they would not fit (default: config)")
    args = ap.parse_args(argv)

    setup_logging(args.report)
    cfg = load_config(args.config, args.set)
    if args.budget_hours is not None:
        cfg["time_budget_hours"] = args.budget_hours
    seed_everything(cfg["seed"])
    t0 = time.time()
    report = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "config": cfg, "argv": list(argv or sys.argv[1:])}
    tracker = Tracker(report, Path(args.report) / "progress.json")
    out = Path(args.out)
    cache_dir = Path(args.cache) if args.cache else out.resolve().parent / "ber_cache"
    ckpt = Checkpointer(cache_dir, args.resume_from, enabled=not args.no_cache,
                        min_free_gb=float(cfg.get("cache", {}).get("min_free_gb", 2.0)))
    log.info("checkpoints: %s%s", cache_dir if ckpt.enabled else "disabled",
             f" (+ read-only {args.resume_from})" if args.resume_from else "")
    try:
        with tracker.stage("load_data"):
            data_dir = find_data_dir(args.data)
            log.info("dataset: %s", data_dir)
            report["data_dir"] = str(data_dir)
            report["env"] = env_info()
            log.info("env: %s", {k: v for k, v in report["env"].items() if v})
            mode = cfg.get("mode", "auto")
            if mode == "auto":
                size = sum(os.path.getsize(Path(data_dir) / sp / f"{sp}_source{k}.tsv")
                           for sp in ("train", "test") for k in (1, 2, 3))
                mode = "scale" if size > cfg.get("scale", {}).get("auto_bytes", 300 * 2**20) else "memory"
            report["mode"] = mode
            log.info("mode: %s", mode)
            if mode == "memory":
                train, test = load_split(data_dir, "train"), load_split(data_dir, "test")
                if train["gold"] is None:
                    raise SystemExit("train_ground_truth.tsv not found")
                s1_order, pool_ids = test["s1_order"], test["pool"]["entity_id"]
        if mode == "memory":
            with tracker.stage("eda_gates"):
                report["eda"] = gates(train, test)
                log.info("EDA: matches/S1 share %s | pool ids in 2+ lists %d | same-country %.4f | pool matched %.3f",
                         {k: round(v, 3) for k, v in report["eda"]["G2_match_count_share"].items()},
                         report["eda"]["G3_pool_ids_in_2plus_lists"], report["eda"]["G5_same_country_rate"],
                         report["eda"]["G4_pool_share_matched"])
            from .pipeline import fit_predict
            pred, cand = fit_predict(train, test, cfg, report, out_dir=args.report, tracker=tracker, ckpt=ckpt)
        else:
            from .scale import fit_predict_scale, test_ids
            pred, cand = fit_predict_scale(data_dir, cfg, report, out_dir=args.report, tracker=tracker, ckpt=ckpt)
            if pred is None:                                # scale.train_only: training data prepared only
                report["progress"]["status"] = "train_only"
                dump_json(report, Path(args.report) / "run_report.json")
                tracker.flush()
                return
            s1_order, pool_ids = test_ids(data_dir)

        with tracker.stage("write_and_validate"):
            mpath, cpath = out / "matching_results.tsv", out / "candidate_pairs.tsv"
            for s, m in pred.items():                           # guard: matches must be candidates
                assert set(m) <= set(cand.get(s, [])), f"match outside candidates for {s}"
            write_ids(mpath, s1_order, pred, "matched_entity_ids")
            write_ids(cpath, s1_order, cand, "candidate_entity_ids")
            report["local_check"] = check_outputs(mpath, cpath, s1_order, pool_ids)
            log.info("local output check: %s", report["local_check"][:5])
            validator = find_validator(data_dir)
            if validator is not None:
                ok, msg = run_official_validator(validator, mpath, cpath, Path(data_dir) / "test")
                report["official_validator"] = {"path": str(validator), "pass": ok, "output": msg[-3000:]}
                log.info("official validator (%s): %s", validator, "PASS" if ok else "FAIL\n" + msg[-2000:])
            else:
                report["official_validator"] = {"path": None, "note": "utils/validate_submission.py not found"}
            report["output_sha256"] = {p.name: sha256(p) for p in (mpath, cpath)}
            report["n_test_s1"] = len(s1_order)
            report["n_test_s1_with_matches"] = sum(1 for v in pred.values() if v)
            report["test_matches_total"] = sum(len(v) for v in pred.values())
        report["runtime_sec"] = round(time.time() - t0, 1)
        report["peak_rss_gb"] = peak_rss_gb()
        with tracker.stage("docs_and_zip"):
            dump_json(report, Path(args.report) / "run_report.json")
            from .docs import write_documentation
            doc_path = Path(args.report) / "Documentation_template.md"
            write_documentation(report, doc_path, team=args.team)
            if args.zip:
                from .package import build_zip
                z = build_zip(args.team, out, CODE_DIR, doc_path, Path(args.zip))
                report["submission_zip"] = str(z)
                log.info("submission zip: %s (%.1f MB)", z, z.stat().st_size / 2**20)
        report["progress"]["status"] = "finished"
        tracker.flush()
        dump_json(report, Path(args.report) / "run_report.json")
        for line in summary_lines(report):
            log.info(line)
        return report
    except BaseException as e:
        report.setdefault("progress", {})["status"] = f"failed: {e!r}"[:500]
        tracker.flush()
        log.error("run failed: %r -- partial metrics in %s; rerun the same command to resume from checkpoints",
                  e, Path(args.report) / "progress.json")
        raise


def summary_lines(r):
    """Compact end-of-run summary for the notebook log."""
    oof, blk, cal = r.get("oof", {}), r.get("blocking", {}), r.get("calibration", {})
    tu, tf = blk.get("train_union", {}), blk.get("train_final", {})
    L = ["=" * 78, "RUN SUMMARY",
         f"  OOF macro F0.5   nested {oof.get('nested_f05', float('nan')):.5f}   final-policy "
         f"{oof.get('macro_f05', float('nan')):.5f}   precision {oof.get('macro_precision_nonempty', float('nan')):.4f}"
         f"   recall {oof.get('macro_recall_matched', float('nan')):.4f}   singleton acc "
         f"{oof.get('singleton_accuracy', float('nan')):.4f}",
         "  OOF by country   " + "  ".join(f"{k}={v['f05']:.4f}(n={v['n']})" for k, v in oof.get("by_country", {}).items()),
         "  OOF by #matches " + "  ".join(f"{k}={v['f05']:.4f}" for k, v in oof.get("by_match_count", {}).items()),
         f"  error budget     {oof.get('error_budget')}",
         f"  blocking (train) PC union {tu.get('PC', float('nan')):.4f} -> final {tf.get('PC', float('nan')):.4f}"
         f"   cand/S1 {tf.get('cand_per_s1_mean', float('nan')):.1f}   RR {tf.get('RR', float('nan')):.6f}",
         f"  models           pass1 AUC {cal.get('pass1_oof', {}).get('auc', float('nan')):.5f}   pass2 AUC "
         f"{cal.get('pass2_oof_raw', {}).get('auc', float('nan')):.5f}   entity AUC "
         f"{cal.get('entity_oof', {}).get('auc', float('nan')):.5f}   ECE after isotonic "
         f"{cal.get('pass2_oof_isotonic', {}).get('ece', float('nan')):.4f}",
         f"  decoder          {r.get('decoder', {}).get('best_config')}   variant {r.get('chosen_variant')}"
         f"   backend {r.get('backend')}   neural {r.get('neural_used')}",
         f"  test             {r.get('n_test_s1')} S1, {r.get('n_test_s1_with_matches')} with matches, "
         f"{r.get('test_matches_total')} matched ids",
         f"  validator        local {r.get('local_check', ['-'])[0]}   official "
         f"{r.get('official_validator', {}).get('pass', 'not found')}",
         f"  sha256           {r.get('output_sha256')}",
         f"  runtime          {fmt_dur(r.get('runtime_sec', 0))}   peak RSS {r.get('peak_rss_gb')} GB   "
         f"checkpoint hits {len(r.get('checkpoints', {}).get('hits', []))}",
         f"  zip              {r.get('submission_zip', '-')}", "=" * 78]
    return L


if __name__ == "__main__":
    main()
