"""Decision layer v3: tuned on the test-like proxy (src/proxy.py), applied to the saved test probabilities.

The pair model (LightGBM pass-2, fold-mean) is not touched; only what happens after its probabilities changes:
  1. calibration fitted on the proxy (fold-mean probabilities at test density) instead of the single-model OOF;
  2. exclusivity as a probability, not a veto. A pool record belongs to at most one S1, so for independent claims
     p_i the probability that S1 i owns it is o_i / (1 + o_i + sum_rivals o_k), o = p / (1 - p). An uncontested
     claim keeps exactly the probability LightGBM gave it; a contested one is lowered by exactly the rival
     evidence (lam scales the rivals; 'hard' = the submitted rule: every non-top claim -> 0);
  3. expected-F0.5 decoding per S1 over its top-0..k sets (Poisson-binomial DP), logit recalibration (a, b);
  4. entity gate h = P(S1 has >= 1 true match), a per-S1 model trained on the proxy: E[F(empty)] = 1 - h;
  5. one owner per pool record in the final output (soft exclusivity may select a record twice).
Every choice is made by nested CV over proxy regions (whole cities), so reported gains are out-of-sample.

    python -m src.decide --n 150000 [--apply]
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .blocking import group_rank
from .cache import Checkpointer
from .compat import effective_cpus
from .decode import best_k_from, build_matrix, decode, expected_f_values, recal, selections
from .models import fit_isotonic
from .proxy import proxy_key, region_folds

log = logging.getLogger("ber")
MODES = ["hard", ("joint", 1.0), ("hardjoint", 1.0), ("joint", 0.5), "none"]
RECAL_A = [1.0, 1.25, 1.5, 2.0]
RECAL_B = [-0.5, -0.25, 0.0, 0.25]
MAX_C = 15
H_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.9,
                bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=7)
H_ROUNDS = 400


# ------------------------------------------------------------------ exclusivity
def claim_index(c, p_raw, s):
    """Dense record index and rank of each claim on its record (1 = strongest)."""
    _, inv = np.unique(c, return_inverse=True)
    return inv, group_rank(inv, p_raw, tie=s)


def apply_excl(p, inv, rank_c, mode):
    """Exclusivity-adjusted probabilities (see module doc)."""
    if mode == "none":
        return p
    if mode == "hard":
        return np.where(rank_c > 1, 0.0, p)
    kind, lam = mode
    o = np.clip(p, 0.0, 1 - 1e-6)
    o = o / (1.0 - o)
    rival = np.bincount(inv, weights=o)[inv] - o
    q = o / (1.0 + o + lam * rival)
    return np.where(rank_c > 1, 0.0, q) if kind == "hardjoint" else q


def one_owner(rows, q, inv):
    """Keep, for every pool record selected by several S1s, only its highest-q selection."""
    if len(rows) == 0:
        return rows
    order = np.lexsort((-q[rows], inv[rows]))
    r = rows[order]
    first = np.r_[True, inv[r][1:] != inv[r][:-1]]
    return np.sort(r[first])


# ------------------------------------------------------------------ entity gate
H_FEATS = ["h_top1", "h_top2", "h_top3", "h_sum", "h_n01", "h_n03", "h_n05", "h_n07", "h_n09", "h_ncand",
           "h_meta_top", "h_meta_max", "h_p1_top", "h_rival_top", "h_rival_n05_top", "h_lost05", "h_entropy"]


def entity_features(s, c, p2, meta, p1, n_s1):
    """Per-S1 aggregates of the pair probabilities, including competition for its best record (label-free)."""
    inv, rank_c = claim_index(c, p2, s)
    p = p2.astype(np.float64)
    # strongest rival claim on the same record (0 when uncontested)
    order = np.lexsort((-p, inv))
    ii = inv[order]
    first = np.r_[True, ii[1:] != ii[:-1]]
    top_p = np.zeros(inv.max() + 1 if len(inv) else 0)
    sec_p = np.zeros_like(top_p)
    top_p[ii[first]] = p[order][first]
    second = np.r_[False, first[:-1] & ~first[1:]]                  # the 2nd row of a record group
    sec_p[ii[second]] = p[order][second]
    rival = np.where(rank_c == 1, sec_p[inv], top_p[inv])
    n05 = np.bincount(inv, weights=(p > 0.5).astype(np.float64))[inv] - (p > 0.5)
    rk = group_rank(s, p, tie=c)
    E = np.zeros((n_s1, len(H_FEATS)), np.float32)
    f = {n: i for i, n in enumerate(H_FEATS)}
    for r, name in ((1, "h_top1"), (2, "h_top2"), (3, "h_top3")):
        m = rk == r
        E[s[m], f[name]] = p[m]
    top = rk == 1
    E[s[top], f["h_meta_top"]] = meta[top]
    E[s[top], f["h_p1_top"]] = p1[top]
    E[s[top], f["h_rival_top"]] = rival[top]
    E[s[top], f["h_rival_n05_top"]] = n05[top]
    for name, w in (("h_sum", p), ("h_n01", p > 0.1), ("h_n03", p > 0.3), ("h_n05", p > 0.5), ("h_n07", p > 0.7),
                    ("h_n09", p > 0.9), ("h_ncand", np.ones_like(p)), ("h_lost05", (p > 0.5) & (rival > p)),
                    ("h_entropy", -p * np.log(np.clip(p, 1e-9, 1)))):
        E[:, f[name]] = np.bincount(s, weights=np.asarray(w, np.float64), minlength=n_s1)
    mm = np.full(n_s1, 0.0)
    np.maximum.at(mm, s, meta.astype(np.float64))
    E[:, f["h_meta_max"]] = mm
    return E


def fit_h(E, label, folds=None):
    """OOF h over folds (when given) and the model on all rows."""
    import lightgbm as lgb
    prm = dict(H_PARAMS, num_threads=effective_cpus())
    oof = None
    if folds is not None:
        oof = np.zeros(len(E))
        for f in np.unique(folds):
            tr = folds != f
            b = lgb.train(prm, lgb.Dataset(E[tr], label[tr], feature_name=H_FEATS), H_ROUNDS)
            oof[~tr] = b.predict(E[~tr])
    full = lgb.train(prm, lgb.Dataset(E, label, feature_name=H_FEATS), H_ROUNDS)
    return oof, full


def profile(s, c, p):
    """Label-free shape of a set of pair probabilities, comparable between the proxy and test."""
    p = np.asarray(p, np.float64)
    n_s1 = len(np.unique(s))
    strong = p > 0.5
    _, inv = np.unique(c[strong], return_inverse=True)
    claims = np.bincount(inv) if len(inv) else np.zeros(0)
    top = np.zeros(int(s.max()) + 1 if len(s) else 0)
    np.maximum.at(top, s, p)
    return {"pairs_per_s1": len(p) / max(n_s1, 1), "gray_0.2_0.8": float(((p > 0.2) & (p < 0.8)).mean()),
            "strong_per_s1": float(strong.sum() / max(n_s1, 1)),
            "s1_top_below_0.5": float((top[np.unique(s)] < 0.5).mean()),
            "records_claimed_by_2plus_s1": float((claims > 1).mean()) if len(claims) else 0.0}


# ------------------------------------------------------------------ test-shift correction (label-free)
# At equal pool density the test has the same confident (p > 0.9) candidates per S1 as the proxy but 1.3-2x the
# mid-band (0.5-0.9) candidates: extra look-alike records, not extra matches. Keeping the proxy's TRUE pairs per S1
# in every probability band and treating the extra test mass as negatives gives the test's true rate per band.
EDGES = np.r_[0.0, 1.0 / (1.0 + np.exp(-np.linspace(-5.0, 8.0, 40))), 1.0 + 1e-9]


def _bands(p):
    return np.clip(np.searchsorted(EDGES, p, side="right") - 1, 0, len(EDGES) - 2)


def shift_calibrator(p_proxy, y_proxy, n_proxy_s1, p_test, n_test_s1):
    """Monotone map raw p -> P(true) on test under 'true pairs per S1 per band as on the proxy'. Never above the
    proxy's own band rate (the correction only lowers). Returns (isotonic model, per-band table)."""
    from sklearn.isotonic import IsotonicRegression
    nb = len(EDGES) - 1
    bp, bt = _bands(p_proxy), _bands(p_test)
    cnt_p = np.bincount(bp, minlength=nb).astype(np.float64)
    pos_p = np.bincount(bp, weights=np.asarray(y_proxy, np.float64), minlength=nb)
    cnt_t = np.bincount(bt, minlength=nb).astype(np.float64)
    mean_t = np.bincount(bt, weights=p_test, minlength=nb) / np.maximum(cnt_t, 1)
    rate_p = pos_p / np.maximum(cnt_p, 1)
    true_t = pos_p / n_proxy_s1 * n_test_s1                  # expected true test pairs in the band
    rate_t = np.minimum(rate_p, true_t / np.maximum(cnt_t, 1))
    ok = (cnt_t > 0) & (cnt_p > 0)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0, increasing=True)
    iso.fit(mean_t[ok], rate_t[ok], sample_weight=cnt_t[ok])
    table = [{"p_lo": float(EDGES[i]), "p_hi": float(min(EDGES[i + 1], 1.0)),
              "proxy_per_s1": float(cnt_p[i] / n_proxy_s1), "test_per_s1": float(cnt_t[i] / n_test_s1),
              "proxy_true_rate": float(rate_p[i]), "test_true_rate_est": float(rate_t[i])}
             for i in range(nb) if cnt_t[i] > 0 or cnt_p[i] > 0]
    return iso, table


# ------------------------------------------------------------------ evaluation
def f05_per_s1(rows, s, y, G, n_s1):
    tp = np.bincount(s[rows], weights=y[rows].astype(np.float64), minlength=n_s1)
    npred = np.bincount(s[rows], minlength=n_s1).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(tp > 0, 1.25 * tp / (npred + 0.25 * G), 0.0)
    return np.where(G > 0, f, (npred == 0).astype(np.float64))


def select_rows(k, IDX):
    mask = np.arange(IDX.shape[1])[None, :] < k[:, None]
    return IDX[mask]


def run_policy(s, c, p_raw, iso, mode, a, b, h, n_s1, inv=None, rank_c=None, p_cal=None):
    """Selected pair rows for one policy (calibrate -> exclusivity -> recal -> expected-F0.5 -> one owner).
    p_cal: calibrated probabilities to use instead of iso.predict(p_raw) (the shift-corrected variant)."""
    if inv is None:
        inv, rank_c = claim_index(c, p_raw, s)
    q = apply_excl(iso.predict(p_raw) if p_cal is None else p_cal, inv, rank_c, mode)
    P, IDX, _, NV = build_matrix(s, q, n_s1, MAX_C, tie=p_raw)
    k = best_k_from(*expected_f_values(recal(P, a, b), NV), h)
    return one_owner(select_rows(k, IDX), q, inv), q


def grid_scores(s, c, p_raw, y, G, n_s1, iso, h, folds, n_folds, inv, rank_c):
    """Per-fold F0.5 sums of every config (mode, a, b, use_h)."""
    out = {}
    p_cal = iso.predict(p_raw)
    for mode in MODES:
        q = apply_excl(p_cal, inv, rank_c, mode)
        P, IDX, _, NV = build_matrix(s, q, n_s1, MAX_C, tie=p_raw)
        for a in RECAL_A:
            for b in RECAL_B:
                vals, p0 = expected_f_values(recal(P, a, b), NV)
                for uh in (False, True):
                    k = best_k_from(vals, p0, h if uh else None)
                    rows = one_owner(select_rows(k, IDX), q, inv)
                    f = f05_per_s1(rows, s, y, G, n_s1)
                    out[(str(mode), a, b, uh)] = np.bincount(folds, weights=f, minlength=n_folds)
    return out


def nested(S_by_fold, n_folds, n_per_fold, allowed):
    """Nested CV: for each outer fold the config chosen on the other folds (only among `allowed`)."""
    cs = [k for k in S_by_fold[0] if allowed(k)]
    tot, chosen = 0.0, []
    for f in range(n_folds):
        inner = S_by_fold[f]
        best = max(cs, key=lambda k: (inner[k].sum() - inner[k][f]))
        tot += S_by_fold[f][best][f]
        chosen.append(best)
    return tot / n_per_fold.sum(), chosen


# ------------------------------------------------------------------ error tree
def error_tree(rows, s, c, y, G, n_s1, cty, true_union_rank, cap):
    """Plan-v3 error-analysis tree on the proxy: where the F0.5 points are lost."""
    f = f05_per_s1(rows, s, y, G, n_s1)
    tp = np.bincount(s[rows], weights=y[rows].astype(np.float64), minlength=n_s1)
    npred = np.bincount(s[rows], minlength=n_s1).astype(np.float64)
    fp = npred - tp
    ycand = np.bincount(s, weights=y.astype(np.float64), minlength=n_s1)        # true pairs among candidates
    owner = pd.Series(s[y], index=c[y])
    owner = owner[~owner.index.duplicated()]
    frow = rows[~y[rows]]
    other = pd.Series(c[frow]).map(owner).notna().to_numpy()
    n = float(n_s1)

    def F(tp_, np_, G_):
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.where(tp_ > 0, 1.25 * tp_ / (np_ + 0.25 * G_), 0.0)
        return np.where(G_ > 0, v, (np_ == 0).astype(np.float64))

    single = G == 0
    tree = {
        "macro_f05": float(f.mean()),
        "by_country": {k: float(f[cty == k].mean()) for k in np.unique(cty)},
        "singletons": {"share": float(single.mean()), "accuracy": float(f[single].mean()) if single.any() else None,
                       "loss_points": float((1 - f[single]).sum() / n)},
        "missed_entity": {"n": int(((G > 0) & (npred == 0)).sum()),
                          "loss_points_no_true_candidate": float((((G > 0) & (npred == 0) & (ycand == 0))).sum() / n),
                          "loss_points_true_candidate_not_taken": float(((G > 0) & (npred == 0) & (ycand > 0)).sum() / n)},
        # counterfactual gains of fixing one error type in S1s that have gold matches
        "gain_if_no_false_positives": float((F(tp, tp, G) - f)[~single].sum() / n),
        "gain_if_all_true_candidates_taken": float((F(ycand, npred - tp + ycand, G) - f)[~single].sum() / n),
        # upper bound for blocking work: every gold record missing from the candidates added and taken
        "gain_bound_if_missing_gold_became_candidates_and_taken": float(
            (F(tp + (G - ycand), npred + (G - ycand), G) - f)[~single].sum() / n),
        "false_positives": {"total": int(fp.sum()), "record_owned_by_another_s1": int(other.sum()),
                            "unowned_record": int((~other).sum()),
                            "in_singleton_s1": int(fp[single].sum())},
        "count_error_(pred-gold)_for_s1_with_matches": pd.Series(np.clip(npred - G, -3, 3)[~single]).value_counts(
            normalize=True).sort_index().round(4).to_dict(),
        "blocking_recall": {"union": float(len(true_union_rank) / G.sum()),
                            f"at_cap{cap}": float((true_union_rank <= cap).sum() / G.sum()),
                            "at_cap40": float((true_union_rank <= 40).sum() / G.sum()),
                            "candidates": float(y.sum() / G.sum())},
    }
    return tree


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[3]
    ap.add_argument("--cache", default=str(root / "ber_cache"))
    ap.add_argument("--data", default=str(root / "student_resource" / "dataset"))
    ap.add_argument("--report", default=str(Path(__file__).resolve().parents[1] / "reports"))
    ap.add_argument("--out", default=None, help="default: output/v3 or output/v3_shift")
    ap.add_argument("--n", type=int, default=150000)
    ap.add_argument("--apply", action="store_true", help="decode the saved test probabilities and write output")
    ap.add_argument("--min-gain", dest="min_gain", type=float, default=0.0005,
                    help="nested F0.5 a decision step must add to be kept")
    ap.add_argument("--reuse", action="store_true", help="take the tuned policy from decide_report.json (no grid)")
    ap.add_argument("--variant", choices=["v3", "shift"], default="v3",
                    help="shift = also correct the test calibration for its extra mid-band candidates")
    ap.add_argument("--big", action="store_true", help="use the bigtrain models (proxy + test scored with them)")
    ap.add_argument("--fast", action="store_true", help="grid: hard exclusivity, no entity gate (v3 showed the rest "
                                                         "adds nothing)")
    ap.add_argument("--stack", action="store_true", help="second-stage stacker (group + competition context, "
                                                          "optional cross-encoder) trained on the holdout")
    ap.add_argument("--ce", default=None, help="ce_outputs.zip (or folder) from the Kaggle cross-encoder kit")
    ap.add_argument("--tag", default=None, help="suffix of the report folder (keeps variants apart)")
    ap.add_argument("--france-as", dest="france_as", default=None, help="shift variant: apply this country's test "
                                                                        "correction to France (e.g. us)")
    ap.add_argument("--logit-shift", dest="logit_shift", default=None,
                    help="per-country logit offset on calibrated test probabilities, e.g. france:0.25 or all:0.2")
    a = ap.parse_args(argv)
    if a.out is None:
        name = ("big_" if a.big else "") + ("v3" if a.variant == "v3" else "v3_shift")
        a.out = str(Path(__file__).resolve().parents[3] / "output" / name)
    if a.fast:
        MODES[:] = ["hard"]
    from .run import setup_logging
    from .track import dump_json
    rdir = Path(a.report) / (("decide_big" if a.big else "decide") + (f"_{a.tag}" if a.tag else ""))
    setup_logging(rdir)
    t0 = time.time()
    ck = Checkpointer(a.cache)
    run_rep = json.loads((Path(a.report) / "run_report.json").read_text(encoding="utf-8"))
    a.big_keys = None
    tag = None
    if a.big:
        man = json.loads((Path(a.report) / "bigtrain" / "manifest.json").read_text(encoding="utf-8"))
        tag = man["pass2"]
        tk = Path(a.report) / "bigtrain" / "test_keys.json"          # written by `bigtrain test`
        if tk.exists():
            a.big_keys = json.loads(tk.read_text(encoding="utf-8"))
        elif a.apply or a.reuse:
            raise SystemExit("big test scores missing: python -m src.bigtrain test")
    P = ck.load(proxy_key(a.n, tag=tag))
    if P is None:
        raise SystemExit("proxy not built: python -m src.proxy --n %d" % a.n)
    a.stacker = None
    if a.stack:
        P = stack_holdout(a, P)
    s, c, p_raw, y, G = P["s"], P["c"], P["p2"].astype(np.float64), P["y"], P["G"]
    n_s1 = len(P["ids"])
    folds = region_folds(P)
    n_folds = int(folds.max()) + 1
    n_per_fold = np.bincount(folds, minlength=n_folds).astype(np.float64)
    pair_fold = folds[s]
    inv, rank_c = claim_index(c, p_raw, s)
    rep = {"proxy": {"n_s1": n_s1, "pairs": int(len(s)), "info": P["info"], "folds_s1": n_per_fold.tolist()}}
    pc = P["cty"][s]
    rep["profile_proxy"] = {k: profile(s[pc == k], c[pc == k], p_raw[pc == k]) for k in np.unique(P["cty"])}
    log.info("proxy profile: %s", rep["profile_proxy"])

    # ---- 0. the submitted policy, measured on the proxy (runs that stopped at train_only have none)
    cap = int(run_rep["config"]["meta"]["cap"])
    fitted_old = submitted_policy(a, run_rep)
    if fitted_old is not None:
        k_o, IDX_o, _ = decode(s, c, p_raw, np.zeros(n_s1), n_s1, dict(run_rep["config"]["decoder"], use_h=False),
                               fitted_old)
        rows_old = np.array([r for rr in selections(k_o, IDX_o) for r in rr], np.int64)
        rep["submitted_policy_on_proxy"] = error_tree(rows_old, s, c, y, G, n_s1, P["cty"],
                                                      P["true_union_meta_rank"], cap)
        log.info("submitted policy on the proxy: macro F0.5 %.5f %s", rep["submitted_policy_on_proxy"]["macro_f05"],
                 rep["submitted_policy_on_proxy"]["by_country"])

    E = entity_features(s, c, p_raw, P["meta"], P["p1"], n_s1)
    if a.reuse:                                   # policy already tuned: apply it (optionally shift-corrected)
        old = json.loads((rdir / "decide_report.json").read_text(encoding="utf-8"))
        fc = old["final_config"]
        best = (fc["mode"], fc["a"], fc["b"], fc["use_h"])
        mode = next(m for m in MODES if str(m) == best[0])
        iso_all = fit_isotonic(p_raw, y.astype(np.float64))
        key = "test" if a.variant == "v3" else "test_shift"
        old[key] = apply_to_test(a, ck, run_rep, iso_all, mode, best, E_fit=(E, (G > 0).astype(np.float64)),
                                 rdir=rdir, proxy=P)
        dump_json(old, rdir / "decide_report.json")
        log.info("done in %.0fs -> %s", time.time() - t0, rdir / "decide_report.json")
        return

    # ---- entity gate: OOF h over region folds
    h_oof, _ = fit_h(E, (G > 0).astype(np.float64), folds)
    from sklearn.metrics import roc_auc_score
    rep["entity_gate"] = {"auc_oof": float(roc_auc_score(G > 0, h_oof)), "singleton_share": float((G == 0).mean())}
    log.info("entity gate h: OOF AUC %.4f (singletons %.3f)", rep["entity_gate"]["auc_oof"], (G == 0).mean())

    # ---- nested CV over (calibration fitted without the outer fold) x (mode, a, b, h)
    S_by_fold = []
    for f in range(n_folds):
        tr = pair_fold != f
        iso_f = fit_isotonic(p_raw[tr], y[tr].astype(np.float64))
        S_by_fold.append(grid_scores(s, c, p_raw, y, G, n_s1, iso_f, h_oof, folds, n_folds, inv, rank_c))
        log.info("  grid for outer fold %d/%d done (%.0fs)", f + 1, n_folds, time.time() - t0)
    iso_all = fit_isotonic(p_raw, y.astype(np.float64))
    S_all = grid_scores(s, c, p_raw, y, G, n_s1, iso_all, h_oof, folds, n_folds, inv, rank_c)
    steps = {
        "1_recalibrated_on_proxy_hard_exclusivity": lambda k: k[0] == "hard" and not k[3],
        "2_plus_exclusivity_as_probability": lambda k: not k[3],
        "3_plus_entity_gate_h": lambda k: True,
        "ref_exclusivity_off": lambda k: k[0] == "none",
        "ref_joint_only_no_veto": lambda k: k[0].startswith("('joint'"),
    }
    rep["nested_f05"] = {}
    for name, allowed in list(steps.items()):
        if not any(allowed(k) for k in S_all):               # e.g. --fast leaves the reference modes empty
            steps.pop(name)
            continue
        v, chosen = nested(S_by_fold, n_folds, n_per_fold, allowed)
        rep["nested_f05"][name] = {"f05": v, "chosen_per_fold": [list(x) for x in chosen]}
        log.info("nested F0.5 %-45s %.5f  chosen %s", name, v, chosen[0])
    # a step is kept only if it adds >= min_gain nested F0.5 over the previous kept step
    nf = rep["nested_f05"]
    use = "1_recalibrated_on_proxy_hard_exclusivity"
    for nxt in ("2_plus_exclusivity_as_probability", "3_plus_entity_gate_h"):
        if nf[nxt]["f05"] - nf[use]["f05"] >= a.min_gain:
            use = nxt
    rep["accepted_step"] = use
    best = max((k for k in S_all if steps[use](k)), key=lambda k: S_all[k].sum())
    mode = next(m for m in MODES if str(m) == best[0])
    rep["final_config"] = {"mode": best[0], "a": best[1], "b": best[2], "use_h": best[3],
                           "in_sample_f05": float(S_all[best].sum() / n_s1)}
    rep["top_configs"] = [[str(list(k)), float(v.sum() / n_s1)] for k, v in
                          sorted(S_all.items(), key=lambda kv: -kv[1].sum())[:15]]
    rows_new, _ = run_policy(s, c, p_raw, iso_all, mode, best[1], best[2], h_oof if best[3] else None, n_s1, inv,
                             rank_c)
    rep["new_policy_on_proxy_in_sample"] = error_tree(rows_new, s, c, y, G, n_s1, P["cty"], P["true_union_meta_rank"],
                                                      cap)
    log.info("final policy %s | in-sample proxy F0.5 %.5f", rep["final_config"],
             rep["new_policy_on_proxy_in_sample"]["macro_f05"])
    dump_json(rep, rdir / "decide_report.json")

    if a.apply:
        rep["test" if a.variant == "v3" else "test_shift"] = apply_to_test(
            a, ck, run_rep, iso_all, mode, best, E_fit=(E, (G > 0).astype(np.float64)), rdir=rdir, proxy=P)
        dump_json(rep, rdir / "decide_report.json")
    log.info("done in %.0fs -> %s", time.time() - t0, rdir / "decide_report.json")


def proxy_pair_ids(P, data, cache):
    """String ids (S1, candidate) of every proxy pair (partition order of src.proxy.build_proxy)."""
    import pyarrow.parquet as pq

    from .scale import normalized_source
    d = Path(data)
    tr = {k: normalized_source(d / "train" / f"train_source{k}.tsv", Path(cache)) for k in (2, 3)}
    cand, off = np.empty(len(P["c"]), dtype=object), 0
    for part in sorted(set(P["cty"])):
        pid = np.concatenate([pq.read_table(tr[k], columns=["id"], filters=[("cty", "==", part)]).column(0)
                              .to_numpy(zero_copy_only=False) for k in (2, 3)])
        m = (P["c"] >= off) & (P["c"] < off + len(pid))
        cand[m] = pid[P["c"][m] - off]
        off += len(pid)
    return P["ids"][P["s"]], cand


def stack_holdout(a, P):
    """Fit the second-stage stacker on the holdout (city folds); the holdout's p2 becomes its OOF output, so the
    decoder is tuned on honest stacked probabilities. The fitted model is kept in a.stacker for test."""
    from sklearn.metrics import log_loss, roc_auc_score

    from .stack import attach_ce, fit_stacker, load_ce, stack_features
    ce = load_ce(a.ce) if a.ce else {}
    use_ce = "holdout" in ce and "test" in ce                  # a feature must exist on both sides
    ce_h = attach_ce(*proxy_pair_ids(P, a.data, a.cache), ce["holdout"]) if use_ce else None
    X, names = stack_features(P["s"], P["c"], P["p2"], P["p1"], P["meta"], ce_h)
    y = P["y"].astype(np.float64)
    oof, model = fit_stacker(X, y, region_folds(P)[P["s"]], names)
    pb = P["p2"].astype(np.float64)
    log.info("stacker on the holdout (OOF): AUC %.6f -> %.6f | logloss %.5f -> %.5f", roc_auc_score(y, pb),
             roc_auc_score(y, oof), log_loss(y, np.clip(pb, 1e-7, 1 - 1e-7)), log_loss(y, np.clip(oof, 1e-7, 1 - 1e-7)))
    a.stacker = (model, names, ce["test"] if use_ce else None)
    P = dict(P)
    P["p2"] = oof.astype(np.float32)
    return P


def submitted_policy(a, run_rep):
    """Calibrator + decoder config of a run that went through fit_predict_scale's decoder stage, else None."""
    op = Path(a.report) / "oof_pairs.parquet"
    if "decoder" not in run_rep or not op.exists():
        return None
    oof = pd.read_parquet(op, columns=["p1", "y"])
    return {"iso": fit_isotonic(oof["p1"].to_numpy(np.float64), oof["y"].to_numpy().astype(np.float64)),
            "iso_h": fit_isotonic(np.r_[0.0, 1.0], np.r_[0.0, 1.0]), "config": run_rep["decoder"]["best_config"]}


def test_keys(run_rep, ck, data_dir, cache):
    """scale-test checkpoint key per test partition; mirrors the key in scale.fit_predict_scale."""
    from .cache import code_hash
    from .proxy import run_keys
    from .scale import normalized_source, partition_keys
    cfg = run_rep["config"]
    d = Path(data_dir)
    paths = {sp: {k: normalized_source(d / sp / f"{sp}_source{k}.tsv", Path(cache)) for k in (1, 2, 3)}
             for sp in ("train", "test")}
    tau = float(run_rep["meta_blocker"]["tau"])
    _, k_p1, k_p2 = run_keys(cfg, paths, ck, int(cfg["scale"]["train_s1"]), tau)
    bcfg = dict(cfg["blocking"])
    bcfg.update(cfg["scale"].get("blocking", {}))
    t = paths["test"]
    keys = {part: ck.key("scale-test", part, str(t[1]), str(t[2]), str(t[3]), k_p1, k_p2, bcfg, tau, cfg["meta"],
                         code_hash("bigblock", "features", "collective"), "v2")
            for part in partition_keys([t[1]])}
    return keys, t


def apply_to_test(a, ck, run_rep, iso, mode, best, E_fit, rdir, proxy=None):
    from .io_utils import check_outputs, find_validator, run_official_validator, write_ids
    from .scale import test_ids
    d = Path(a.data)
    keys, paths = test_keys(run_rep, ck, a.data, a.cache)
    if getattr(a, "big_keys", None):
        keys = a.big_keys
    parts = sorted(keys)
    S, C, PR, M, P1, ids1, idsp = [], [], [], [], [], [], []
    off_s = off_c = 0
    for part in parts:
        k = keys[part]
        saved = ck.load(k)
        if saved is None:
            raise SystemExit(f"test probabilities for {part} not found ({k})")
        s, c, p2, meta, p1 = saved
        i1 = pd.read_parquet(paths[1], columns=["id"], filters=[("cty", "==", part)])["id"].to_numpy(object)
        ip = np.concatenate([pd.read_parquet(paths[j], columns=["id"], filters=[("cty", "==", part)])["id"]
                             .to_numpy(object) for j in (2, 3)])
        prof = profile(s, c, p2)
        log.info("test profile %s: %s", part, prof)
        S.append(s.astype(np.int64) + off_s)
        C.append(c.astype(np.int64) + off_c)
        PR.append(p2.astype(np.float64))
        M.append(meta)
        P1.append(p1)
        ids1.append(i1)
        idsp.append(ip)
        off_s += len(i1)
        off_c += len(ip)
        log.info("test %s: %d S1, %d pool, %d pairs (%s)", part, len(i1), len(ip), len(s), k)
    s, c, p_raw = np.concatenate(S), np.concatenate(C), np.concatenate(PR)
    meta, p1 = np.concatenate(M), np.concatenate(P1)
    n_by_part = [len(x) for x in ids1]
    ids1, idsp = np.concatenate(ids1), np.concatenate(idsp)
    n_s1 = len(ids1)
    if getattr(a, "stacker", None) is not None:              # same second stage as on the holdout
        from .stack import attach_ce, stack_features
        model, names, ce_test = a.stacker
        ce_t = attach_ce(ids1[s], idsp[c], ce_test) if ce_test is not None else None
        Xt, names_t = stack_features(s, c, p_raw, p1, meta, ce_t)
        if names_t != names:
            raise SystemExit(f"stacker features differ between holdout and test: {names} vs {names_t}")
        p_raw = model.predict(Xt)
        PR = [p_raw[lo:lo + len(x)] for lo, x in zip(np.cumsum([0] + [len(x) for x in PR[:-1]]), PR)]
        log.info("test pairs re-scored by the stacker (mean p %.4f)", float(p_raw.mean()))
        del Xt
    # guard: the submitted policy on these arrays must give the submitted number of matches
    fitted_old = submitted_policy(a, run_rep)
    if fitted_old is not None:
        k_old, _, _ = decode(s, c, p_raw, np.zeros(n_s1), n_s1, dict(run_rep["config"]["decoder"], use_h=False),
                             fitted_old)
        if getattr(a, "big_keys", None):                     # new models: nothing to reproduce
            log.info("big models: submitted decoder would select %d matches on the new probabilities",
                     int(k_old.sum()))
        elif int(k_old.sum()) != int(run_rep["test_matches_total"]):
            raise SystemExit(f"test arrays do not reproduce the submission: {int(k_old.sum())} vs "
                             f"{run_rep['test_matches_total']} matches")
        else:
            log.info("test arrays reproduce the submitted %d matches", int(k_old.sum()))
        del k_old
    inv, rank_c = claim_index(c, p_raw, s)
    h = None
    if best[3]:
        E_te = entity_features(s, c, p_raw, meta, p1, n_s1)
        _, hm = fit_h(*E_fit)
        h = hm.predict(E_te)
    p_cal, shift_tables = None, None
    if a.variant == "shift":
        p_cal, shift_tables = np.empty(len(p_raw)), {}
        py, pp, pcty = proxy["y"], proxy["p2"].astype(np.float64), proxy["cty"][proxy["s"]]
        lo, segs, cals = 0, {}, {}
        for part, n_pairs, i1 in zip(parts, [len(x) for x in PR], n_by_part):
            segs[part] = slice(lo, lo + n_pairs)
            m = pcty == part if part in set(proxy["cty"]) else np.ones(len(pp), bool)   # France: pooled proxy
            n_px = int((proxy["cty"] == part).sum()) if part in set(proxy["cty"]) else len(proxy["ids"])
            cals[part], table = shift_calibrator(pp[m], py[m], n_px, p_raw[segs[part]], i1)
            shift_tables[part] = table
            mid = [t for t in table if 0.5 <= t["p_lo"] and t["p_hi"] <= 0.95]
            log.info("shift %s: mid-band true rate proxy %s -> test est %s", part,
                     [round(t["proxy_true_rate"], 2) for t in mid], [round(t["test_true_rate_est"], 2) for t in mid])
            lo += n_pairs
        # a country without labels can borrow the correction of a known country (--france-as us): the pooled
        # estimate reads the transformers' lower confidence on unseen French text as extra look-alikes
        for part, seg in segs.items():
            src = getattr(a, "france_as", None) if part == "france" else None
            if src and src in cals:
                log.info("shift %s: using the %s correction", part, src)
            p_cal[seg] = cals[src if src in cals else part].predict(p_raw[seg])
    if getattr(a, "logit_shift", None):                    # e.g. "france:0.25" or "all:0.2": more (+) / less (-) recall
        if p_cal is None:
            p_cal = iso.predict(p_raw)
        lo = 0
        bounds = {}
        for part, x in zip(parts, PR):
            bounds[part] = slice(lo, lo + len(x))
            lo += len(x)
        for item in a.logit_shift.split(","):
            part, dv = item.split(":")
            targets = parts if part == "all" else [part]
            for t in targets:
                seg = bounds[t]
                q0 = np.clip(p_cal[seg], 1e-7, 1 - 1e-7)
                p_cal[seg] = np.where(p_cal[seg] > 0, 1 / (1 + np.exp(-(np.log(q0 / (1 - q0)) + float(dv)))), 0.0)
                log.info("logit shift %+.2f applied to %s", float(dv), t)
    rows, q = run_policy(s, c, p_raw, iso, mode, best[1], best[2], h, n_s1, inv, rank_c, p_cal=p_cal)
    order = np.lexsort((-q[rows], s[rows]))
    rs = rows[order]
    bnd = np.flatnonzero(np.r_[True, s[rs][1:] != s[rs][:-1], True])
    pred = {ids1[s[rs[x]]]: idsp[c[rs[x:y]]].tolist() for x, y in zip(bnd[:-1], bnd[1:])}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "matching_results.tsv"
    s1_order, pool_ids = test_ids(a.data)
    write_ids(mpath, s1_order, pred, "matched_entity_ids")
    if getattr(a, "big_keys", None):
        # these candidates come from this run's blocking: write them (best meta score first) next to the matches
        cpath = out / "candidate_pairs.tsv"
        o2 = np.lexsort((-meta, s))
        bnd2 = np.flatnonzero(np.r_[True, s[o2][1:] != s[o2][:-1], True])
        cand = {ids1[s[o2[x]]]: idsp[c[o2[x:y]]].tolist() for x, y in zip(bnd2[:-1], bnd2[1:])}
        write_ids(cpath, s1_order, cand, "candidate_entity_ids")
        del cand, o2
    else:
        cpath = Path(a.out).parent / "candidate_pairs.tsv"               # candidates are unchanged
    res = {"pairs_selected": int(len(rows)), "s1_with_matches": int(len(pred)), "n_s1": int(n_s1),
           "mean_matches": float(len(rows) / n_s1), "empty_rate": float(1 - len(pred) / n_s1),
           "path": str(mpath), "candidates_file": str(cpath)}
    res["by_partition"] = {}
    lo = 0
    npred = np.bincount(s[rows], minlength=n_s1)
    for part, i1 in zip(parts, n_by_part):
        seg = npred[lo:lo + i1]
        res["by_partition"][part] = {"n_s1": int(i1), "mean_matches": float(seg.mean()),
                                     "empty_rate": float((seg == 0).mean())}
        lo += i1
    if shift_tables is not None:
        res["shift_tables"] = shift_tables
    res["local_check"] = check_outputs(mpath, cpath, s1_order, pool_ids)
    v = find_validator(a.data)
    if v is not None:
        ok, msg = run_official_validator(v, mpath, cpath, d / "test")
        res["official_validator"] = {"pass": ok, "output": msg[-2000:]}
    import hashlib
    res["sha256"] = hashlib.sha256(mpath.read_bytes()).hexdigest()
    log.info("test decoded: %s", {k: v for k, v in res.items() if k not in ("local_check", "shift_tables")})
    return res


if __name__ == "__main__":
    main()
