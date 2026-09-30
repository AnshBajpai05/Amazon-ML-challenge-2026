"""Validation beyond out-of-fold (plan section 9).

    python -m src.evaluate --data DATA --loco [--density]      # leave-one-country-out (+ half-pool check)
    python -m src.evaluate --score output/matching_results.tsv --gold GOLD.tsv [--cand output/candidate_pairs.tsv]

LOCO trains on every country but one and scores the held-out one with the full pipeline
(blocking, IDF, thresholds all refitted) - the only honest proxy for the unseen France partition.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from .io_utils import find_data_dir, load_gold, load_split, read_tsv
from .metrics import blocking_kpis, evaluate
from .normalize import cty_key
from .cache import Checkpointer
from .run import CODE_DIR, load_config, seed_everything, setup_logging
from .track import Tracker, dump_json

log = logging.getLogger("ber")


def _subset(split, s1_mask, pool_mask):
    s1 = split["s1"][s1_mask].reset_index(drop=True)
    pool = split["pool"][pool_mask].reset_index(drop=True)
    pids = set(pool["entity_id"])
    gold = {s: [x for x in split["gold"].get(s, []) if x in pids] for s in s1["entity_id"]}
    return {"s1": s1, "pool": pool, "gold": gold, "s1_order": s1["entity_id"].tolist()}


def loco(data_dir, cfg, density=False, min_s1=100, ckpt=None, report_dir=None):
    tr = load_split(data_dir, "train")
    c1 = np.array([cty_key(x) for x in tr["s1"]["country"]], dtype=object)
    cp = np.array([cty_key(x) for x in tr["pool"]["country"]], dtype=object)
    out = {}
    for c in sorted(set(c1)):
        m1, mp = c1 == c, cp == c
        if m1.sum() < min_s1 or (~m1).sum() < min_s1:
            log.info("LOCO: skip %s (too few S1 records on one side)", c)
            continue
        train, test = _subset(tr, ~m1, ~mp), _subset(tr, m1, mp)
        from .pipeline import fit_predict
        t0 = time.time()
        rep = {}
        tr_ = Tracker(rep, Path(report_dir) / f"loco_progress_{c}.json" if report_dir else None, prefix=f"loco[{c}] ")
        fit_predict(train, test, cfg, rep, test_gold=test["gold"], tracker=tr_, ckpt=ckpt)
        res = {"heldout": rep["heldout"], "heldout_blocking": rep["heldout_blocking"],
               "train_nested_oof_f05": rep["oof"]["nested_f05"], "decoder": rep["decoder"]["best_config"],
               "seconds": round(time.time() - t0, 1)}
        if density:                                        # thresholds must not depend on one pool size
            matched = {x for v in test["gold"].values() for x in v}
            rng = np.random.RandomState(cfg["seed"])
            ids = test["pool"]["entity_id"].to_numpy(object)
            keep = np.array([(i in matched) or (rng.rand() >= 0.5) for i in ids])
            thin = {**test, "pool": test["pool"][keep].reset_index(drop=True)}
            rep2 = {}
            fit_predict(train, thin, cfg, rep2, test_gold=test["gold"], tracker=Tracker(rep2, prefix=f"half-pool[{c}] "),
                        ckpt=ckpt)
            res["half_pool_f05"] = rep2["heldout"]["macro_f05"]
        out[f"train_without_{c} -> score_{c}"] = res
        if report_dir:
            dump_json(out, Path(report_dir) / "loco.json")          # saved after every country
        log.info("LOCO %s: F0.5 = %.5f (PC %.4f)", c, res["heldout"]["macro_f05"], res["heldout_blocking"]["PC"])
    return out


def score_file(match_path, gold_path, cand_path=None, test_dir=None):
    gold = load_gold(gold_path)
    d = read_tsv(match_path)
    pred = {s: [x for x in m.split(",") if x] for s, m in zip(d.iloc[:, 0], d.iloc[:, 1])}
    cand = None
    if cand_path:
        c = read_tsv(cand_path)
        cand = {s: [x for x in m.split(",") if x] for s, m in zip(c.iloc[:, 0], c.iloc[:, 1])}
    country = None
    if test_dir:
        s1 = read_tsv(Path(test_dir) / "test_source1.tsv")
        country = dict(zip(s1["entity_id"], (cty_key(x) for x in s1["country"])))
    res = evaluate(pred, gold, country, cand)
    if cand is not None:
        n_pool = sum(len(read_tsv(Path(test_dir) / f"test_source{k}.tsv")) for k in (2, 3)) if test_dir else 1
        res["blocking"] = blocking_kpis(cand, gold, n_pool, country)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/kaggle/input")
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--report", default=str(CODE_DIR / "reports"))
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--loco", action="store_true")
    ap.add_argument("--density", action="store_true")
    ap.add_argument("--cache", default=None, help="checkpoint dir (default: <report>/../../ber_cache)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--score", default=None)
    ap.add_argument("--gold", default=None)
    ap.add_argument("--cand", default=None)
    ap.add_argument("--test-dir", default=None)
    a = ap.parse_args(argv)
    setup_logging(a.report)
    if a.score:
        res = score_file(a.score, a.gold, a.cand, a.test_dir)
        dump_json(res, Path(a.report) / "score.json")
        log.info("macro F0.5 = %.5f  %s", res["macro_f05"], {k: v for k, v in res.items() if k.startswith("by_")})
        return res
    cfg = load_config(a.config, a.set)
    seed_everything(cfg["seed"])
    ck = Checkpointer(a.cache or Path(a.report).resolve().parent.parent / "ber_cache", enabled=not a.no_cache)
    res = loco(find_data_dir(a.data), cfg, a.density, ckpt=ck, report_dir=a.report)
    dump_json(res, Path(a.report) / "loco.json")
    rp = Path(a.report) / "run_report.json"
    if rp.exists():                                         # refresh the documentation with the LOCO table
        import json
        from .docs import write_documentation
        write_documentation(json.load(open(rp, encoding="utf-8")), Path(a.report) / "Documentation_template.md",
                            loco=res)
    return res


if __name__ == "__main__":
    main()
