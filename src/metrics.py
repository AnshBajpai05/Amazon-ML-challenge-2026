"""Metric, breakdowns, error budget (plan 2, 9, A.2, A.3) and blocking KPIs (plan 5)."""
from __future__ import annotations

from collections import Counter

import numpy as np


def f05_entity(pred, gold):
    """Per-S1 F0.5 exactly as defined, including the singleton rule."""
    pred, gold = set(pred), set(gold)
    if not gold:
        return 1.0 if not pred else 0.0
    tp = len(pred & gold)
    return 0.0 if tp == 0 else 1.25 * tp / (len(pred) + 0.25 * len(gold))


def macro_f05(pred_map, gold_map):
    """Average over EVERY S1 in gold_map (entities with no prediction count as empty)."""
    if not gold_map:
        return float("nan")
    return float(np.mean([f05_entity(pred_map.get(s, ()), g) for s, g in gold_map.items()]))


def error_budget(pred, gold, cand):
    """Where the macro-F0.5 points go; the parts sum exactly to 1 - macro F0.5."""
    lost, n = Counter(), len(gold)
    for s, g in gold.items():
        p, c, g = set(pred.get(s, ())), set(cand.get(s, ())), set(g)
        loss = 1 - f05_entity(p, g)
        if loss == 0:
            continue
        if not g:
            lost["singleton_false_merge"] += loss
        elif not p:
            lost["missed_entity_" + ("blocking" if not (g & c) else "model")] += loss
        elif not (p & g):
            lost["wrong_pick_only"] += loss
        else:
            fp_free = f05_entity(p & g, g)
            lost["extra_false_positives"] += fp_free - f05_entity(p, g)
            lost["missed_extra_matches"] += 1 - fp_free
    return {k: round(v / max(n, 1), 5) for k, v in lost.most_common()}


def evaluate(pred, gold, country=None, cand=None):
    """Overall / per-country / per-bucket macro F0.5, macro P/R, singleton accuracy, error budget."""
    ids = list(gold)
    f = np.array([f05_entity(pred.get(s, ()), gold[s]) for s in ids])
    ng = np.array([len(gold[s]) for s in ids])
    npred = np.array([len(set(pred.get(s, ()))) for s in ids])
    tp = np.array([len(set(pred.get(s, ())) & set(gold[s])) for s in ids])
    prec = tp[npred > 0] / npred[npred > 0]
    rec = tp[ng > 0] / ng[ng > 0]
    out = {"n_s1": len(ids), "macro_f05": float(f.mean()) if len(f) else float("nan"),
           "macro_precision_nonempty": float(prec.mean()) if len(prec) else float("nan"),
           "macro_recall_matched": float(rec.mean()) if len(rec) else float("nan"),
           "singleton_accuracy": float((npred[ng == 0] == 0).mean()) if (ng == 0).any() else float("nan"),
           "matched_entity_nonempty_rate": float((npred[ng > 0] > 0).mean()) if (ng > 0).any() else float("nan")}
    buckets = np.minimum(ng, 3)
    out["by_match_count"] = {("3+" if b == 3 else str(b)): {"n": int((buckets == b).sum()),
                                                           "f05": float(f[buckets == b].mean())}
                             for b in range(4) if (buckets == b).any()}
    if country is not None:
        c = np.array([country.get(s, "?") for s in ids], dtype=object)
        out["by_country"] = {k: {"n": int((c == k).sum()), "f05": float(f[c == k].mean())} for k in sorted(set(c))}
    if cand is not None:
        out["error_budget"] = error_budget(pred, gold, cand)
    return out


def blocking_kpis(cand, gold, n_pool, country=None, channel_hits=None):
    """PC (recall ceiling), entity completeness, RR, PQ, candidates per S1 (+ per country)."""
    def one(ids):
        tot_true = sum(len(gold[s]) for s in ids)
        found = sum(len(set(cand.get(s, ())) & set(gold[s])) for s in ids)
        ncand = np.array([len(cand.get(s, ())) for s in ids]) if ids else np.zeros(1)
        matched = [s for s in ids if gold[s]]
        complete = sum(set(gold[s]) <= set(cand.get(s, ())) for s in matched)
        return {"n_s1": len(ids), "true_pairs": int(tot_true),
                "PC": found / tot_true if tot_true else float("nan"),
                "entity_completeness": complete / len(matched) if matched else float("nan"),
                "RR": 1 - ncand.sum() / max(len(ids) * n_pool, 1),
                "PQ": found / max(ncand.sum(), 1),
                "cand_per_s1_mean": float(ncand.mean()), "cand_per_s1_p95": float(np.percentile(ncand, 95)),
                "cand_per_s1_max": int(ncand.max())}
    ids = list(gold)
    out = one(ids)
    if country is not None:
        out["by_country"] = {k: one([s for s in ids if country.get(s) == k])
                             for k in sorted(set(country.get(s, "?") for s in ids))}
    if channel_hits is not None:
        out["unique_true_pairs_by_channel"] = channel_hits
    return out
