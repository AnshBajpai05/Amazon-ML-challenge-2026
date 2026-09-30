"""Write the methodology document (plan section 15) from run_report.json (+ loco.json if present).

    python -m src.docs --report reports            # regenerate reports/Documentation_template.md
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def f(x, d=4):
    if x is None:
        return "–"
    if isinstance(x, float):
        return "–" if math.isnan(x) else f"{x:.{d}f}"
    return str(x)


def _mode(rep):
    mode = rep.get("config", {}).get("partition", "auto")
    if mode == "auto":
        mode = "country" if rep.get("same_country_match_rate", 1.0) >= 0.999 else "none"
    return mode


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(f(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


DIAGRAM = """```
train/ & test/ TSVs -> safe read (sep=\\t, dtype=str, keep_default_na=False, QUOTE_NONE, UTF-8)
      |
[0] NORMALIZE ONCE  fold accents . canonical short forms . legal/DBA/landmark/number split . skeleton keys
      |
[1] BLOCKING  (per country partition; TF-IDF/IDF fitted on the corpus being searched, label-free)
      c1 name-char . c2 name-word . c3 skeleton . c4 address-char . c5 name+address
      c6 reverse-kNN (pool->S1) . c7 2-hop (near-duplicates of found records) . [c8 dense, optional]
      +- union with provenance -> learned META-BLOCKER (OOF) -> candidate_pairs.tsv
      |
[2] PAIR FEATURES (~110)  string sims . abbreviation-aware IDF alignment . numbers/postal .
                          legal/DBA/acronym . rarity/chain . provenance
      |
[3] GBDT pass-1 (5 folds grouped by S1, monotone constraints) -> out-of-fold p1
[4] collective features from p1 (ranks both ways, margins, sibling support) [+ cross-encoder] -> pass-2
[5] entity model: h = P(at least one candidate of this S1 is a true match)
      |
[6] DECISION  isotonic calibration -> exclusivity -> per-S1 expected-F0.5 set decoding (nested-CV tuned)
      +-> matching_results.tsv  (subset of candidates) -> validator PASS
```"""

DECISIONS = """1. **Decide per entity, optimizing the metric itself.** F0.5 is computed per S1, so each S1's output set
   maximizes its *expected* F0.5 with an exact Poisson-binomial dynamic programme over top-k sets (verified
   against brute force over all subsets). A single global threshold cannot be optimal: the first pick needs
   p > 0.5 while an extra pick needs p > F_current / 1.25.
2. **A dedicated singleton gate.** "Does this S1 have any match?" is worth a full point on every entity, so it
   gets its own model h; the decoder scores the empty set as 1 − h.
3. **Zero-shot country transfer by construction.** Language-neutral normalization, a generic abbreviation rule,
   skeleton keys and corpus-relative IDF let French records match with zero French labels. Country is only a
   partition key; there is no country one-hot and no country-specific threshold.
4. **Use the structure of the data.** S1 is deduplicated, so a pool record belongs to at most one S1
   (exclusivity). An S1's matches are near-duplicates of each other, which 2-hop blocking and sibling-support
   features exploit.
5. **Audited, learned blocking.** Seven complementary channels feed a learned meta-blocker; recall ceiling,
   reduction ratio and per-channel gains are reported below.
6. **A robust, calibrated core model.** Gradient-boosted trees with monotone constraints, isotonic calibration
   and fold-averaged inference; one fold assignment shared by every learned stage (no stacking leakage).
7. **One command reproduces everything.** Pinned environment, fixed seeds, SHA-256 of both outputs."""


def write_documentation(rep: dict, path, loco=None, team=None):
    cfg = rep.get("config", {})
    eda = rep.get("eda", {})
    blk = rep.get("blocking", {})
    oof = rep.get("oof", {})
    dec = rep.get("decoder", {})
    env = rep.get("env", {})
    L = []
    A = L.append
    A("# Business Entity Resolution — Methodology\n")
    if team:
        A(f"**Team:** {team}  ")
    A("*Generated automatically from `reports/run_report.json` of the run that produced `output/`.*  ")
    A(f"Run started {rep.get('started', '–')}, runtime {f(rep.get('runtime_sec', 0) / 60, 1)} min, "
      f"peak RSS {f(rep.get('peak_rss_gb'), 2)} GB, backend `{rep.get('backend')}`, GPU: {env.get('gpu') or 'none'}.\n")
    A("**Headline (out-of-fold on train, nested CV — decoder tuned on 4 folds, scored on the 5th):** "
      f"macro F0.5 = **{f(oof.get('nested_f05'), 5)}**. "
      f"Public leaderboard: _fill in after upload_.\n")

    A("## 1. Methodology overview\n")
    A(DIAGRAM + "\n")
    A("**Seven design decisions**\n")
    A(DECISIONS + "\n")

    A("## 2. Data understanding (EDA gates)\n")
    rows = [(k, v) for k, v in eda.get("G1_rows_train", {}).items()]
    rows += [(f"test {k}", v) for k, v in eda.get("G1_rows_test", {}).items()]
    A("**G1 — records per source × country**\n")
    A(table(["source | country", "rows"], rows) + "\n")
    g2 = eda.get("G2_match_count_share", {})
    A(table(["Gate", "Result", "Decision it set"], [
        ("G2 matches per S1 (0 / 1 / 2 / 3+)", " / ".join(f(g2.get(k), 3) for k in ("0", "1", "2", "3+")),
         "singleton gate weight; K per channel ≥ 3× p99 match count"),
        ("G2 p99 / max matches per S1", f"{eda.get('G2_match_count_p99_max')}", "decoder keeps top-15 per S1"),
        ("G2 source share of matches", f"{eda.get('G2_source_share_of_matches')}", "source flag feature"),
        ("G3 pool ids in ≥ 2 gold lists", f"{eda.get('G3_pool_ids_in_2plus_lists')}",
         "exclusivity enabled in the decoder grid (γ tuned)"),
        ("G4 share of pool matched to some S1", f(eda.get("G4_pool_share_matched")), "precision focus; reverse kNN"),
        ("G5 matched pairs with same country", f(eda.get("G5_same_country_rate")),
         f"partition mode = `{_mode(rep)}`"),
        ("G6 matched S1 with ≥ 2 matches from one source", f(eda.get("G6_s1_with_2plus_from_same_source")),
         "2-hop blocking + sibling support"),
        ("G8 S1 records sharing a normalized name (chains)", f"{eda.get('G8_s1_names_shared_by_2plus')}",
         "name-frequency features; address decides chains"),
    ]) + "\n")
    A("**G10 — placeholder values ('', NA, N/A, -, None, …)** — read with `keep_default_na=False`, so a business "
      "literally named \"NA\" stays a string:\n")
    A(table(["split | source | column", "count"], list(eda.get("G10_placeholder_counts", {}).items())) + "\n")
    A("**G9 — drift per country (mean lengths)**\n")
    A(table(["split | source | country", "n", "name len", "addr len", "addr empty"],
            [(k, v["n"], v["name_len_mean"], v["addr_len_mean"], v["addr_empty_rate"])
             for k, v in eda.get("G9_drift", {}).items()]) + "\n")

    A("## 3. Normalization (country-agnostic, standard library + rapidfuzz only)\n")
    A("""- **Unicode folding:** NFKC → lowercase → ligatures (œ→oe, æ→ae, ß→ss) → strip accents. No Unidecode (GPL).
- **Canonical short forms** instead of expansions: street/str/saint → `st`; suite/sainte/société → `ste`;
  road → `rd`, avenue/ave → `av`, boulevard/blvd → `bd`, private → `pvt`, limited → `ltd`, chemin → `ch`,
  faubourg → `fbg`, shree/shri/sree → `sri`, first/1st/1er → `1`, … An ambiguous "St" then matches either reading.
- **Apostrophes, acronyms, numbers:** joe's → joe, l'étoile → etoile, M.G. → mg, 12bis → 12 bis, 411 001 → 411001.
- **Names:** DBA / t/a / aka / parenthesised pieces become *variants*; legal forms (US, India, France, generic
  EU list) are split into their own field, trailing everywhere and leading for French forms (SARL …).
- **Addresses:** comma segments; a landmark cue (near, opp, behind, next to, près de, en face de, à côté de …)
  moves the rest of its segment into a *landmark* field; numbers are extracted, 5+ digit tokens are *postal-like*
  (US ZIP, Indian PIN, French code postal) with no country logic.
- **Skeleton keys** for transliteration: lakshmi/laxmi → `lksm`, aggarwal/agrawal → `agrvl`, mohammed/muhammad → `mhmd`.
- **Abbreviation rule** (language-agnostic): `a ~ b` if a is shorter, starts with b's letter, is a subsequence of b,
  and is a prefix of b / ends with b's last letter / has ≥ 3 letters. Accepts rd~road, pvt~private, fbg~faubourg,
  cie~compagnie; rejects st~south, rd~ridge. This is what lets French abbreviations match with no French labels.
- **Lexicon provenance:** every list in `src/normalize.py` (abbreviations, legal forms, landmark cues, stopwords)
  is hand-written generic linguistic knowledge; no gazetteer, registry, geocoder or external dataset is used.

| Input | Output |
|---|---|
| `ABC Traders Pvt. Ltd.` / `A.B.C. TRADERS PRIVATE LIMITED` | both → core `abc traders`, legal `pvt ltd` |
| `Holdco Inc d/b/a Sunrise Dental` | core `holdco sunrise dental`, variants `holdco` / `sunrise dental` |
| `SARL Boulangerie de l'Étoile` | core `boulangerie etoile`, legal `sarl` |
| `Shop No. 5, Near SBI ATM, M.G. Road, Pune 411 001` | core `shop 5 mg rd pune 411001`, landmark `sbi atm`, postal `411001` |
| `45 Bd Saint-Germain, 75005 Paris, en face de la Poste` | core `45 bd st germain 75005 paris`, landmark `poste` |
""")

    A("## 4. Candidate generation / blocking\n")
    K = cfg.get("blocking", {}).get("k", {})
    A(f"Partition key: normalized country label (mode `{_mode(rep)}`; "
      f"train same-country match rate {f(rep.get('same_country_match_rate'))}). TF-IDF vocabularies and IDF are "
      "fitted per partition on S1 ∪ pool of the split being searched (no labels), so France gets its own weights. "
      "Exact sparse top-K (`sparse_dot_topn`, verified identical to brute force), never an approximate index.\n")
    A(table(["#", "Channel", "Representation", "K"], [
        ("c1", "Name, character", "char_wb 3–4-gram TF-IDF, sublinear, max_df 0.2", K.get("c1")),
        ("c2", "Name, word", "canonical core tokens, TF-IDF", K.get("c2")),
        ("c3", "Name, skeleton", "skeleton tokens, TF-IDF", K.get("c3")),
        ("c4", "Address, character", "3–4-gram TF-IDF on the address core", K.get("c4")),
        ("c5", "Name + address", "character n-grams of both", K.get("c5")),
        ("c6", "Reverse kNN", "each pool record → its top S1s on c5", K.get("c6")),
        ("c7", "2-hop", "pool self-kNN (cos ≥ 0.6) of each S1's top-3 c5 hits", K.get("c7")),
        ("c8", "Dense (optional)", "multilingual-e5-small cosine", f"{K.get('c8')} ({'on' if rep.get('neural_used', {}).get('dense_c8') else 'off'})"),
        ("c9", "Reverse name kNN", "each pool record → its top S1s on the name char TF-IDF", K.get("c9")),
    ]) + "\n")
    A(f"S1 records without an address can only be found through their name, so they search c1–c3 "
      f"{cfg.get('blocking', {}).get('empty_addr_k_mult', 2.0)}× deeper; c9 does the same from the pool side "
      "(an address-less pool record inside a crowded name neighbourhood, e.g. chains).\n")
    mb = rep.get("meta_blocker", {})
    mc = cfg.get("meta", {})
    A(f"**Learned meta-blocking** (the last filter, so its output *is* `candidate_pairs.tsv`): a small GBDT on "
      "channel scores/ranks, channel count and exact name/address/joint cosines + postal agreement, trained "
      f"out-of-fold on the shared folds. Keep a pair if it is in its S1's top-{mc.get('top_keep')} by meta score "
      f"or meta ≥ τ = {f(mb.get('tau'))}, capped at {mc.get('cap')} per S1; τ keeps ≥ {mc.get('recall_ratio')} of "
      "the union's true pairs out-of-fold.\n")
    tu, tf_ = blk.get("train_union", {}), blk.get("train_final", {})
    A("**Blocking KPIs (train, labelled)**\n")
    A(table(["Stage", "PC (recall ceiling)", "Entity completeness", "RR", "PQ", "cand/S1 mean", "p95", "max"], [
        ("union of channels", tu.get("PC"), tu.get("entity_completeness"), tu.get("RR"), tu.get("PQ"),
         tu.get("cand_per_s1_mean"), tu.get("cand_per_s1_p95"), tu.get("cand_per_s1_max")),
        ("after meta-blocking (= candidate file)", tf_.get("PC"), tf_.get("entity_completeness"), tf_.get("RR"),
         tf_.get("PQ"), tf_.get("cand_per_s1_mean"), tf_.get("cand_per_s1_p95"), tf_.get("cand_per_s1_max")),
    ]) + "\n")
    A(table(["Country", "PC final", "Entity completeness", "cand/S1"],
            [(k, v.get("PC"), v.get("entity_completeness"), v.get("cand_per_s1_mean"))
             for k, v in tf_.get("by_country", {}).items()]) + "\n")
    A("**Per-channel contribution (train union):** true pairs found, and true pairs found by that channel only.\n")
    A(table(["Channel", "true pairs found", "unique true pairs"],
            [(k, v["true_pairs_found"], v["unique_true_pairs"])
             for k, v in tu.get("unique_true_pairs_by_channel", {}).items()]) + "\n")
    te = blk.get("test_final", {})
    A(f"Test: {te.get('union_pairs')} union pairs → {te.get('pairs')} candidates "
      f"({f(te.get('cand_per_s1_mean'), 2)} per S1, RR {f(te.get('RR'), 6)}).\n")

    A("## 5. Model architecture and feature engineering\n")
    A(table(["Family", "Features"], [
        ("Name strings", "Jaro-Winkler, normalized Levenshtein, ratio, token-sort/set, partial, LCS, prefix, on core / full / raw names; char, word and skeleton TF-IDF cosines"),
        ("Soft alignment (key)", "IDF-weighted greedy one-to-one token alignment with abbreviation/typo/skeleton-aware token similarity: soft Dice, coverage both ways, **max IDF of an unmatched token on each side**, shared-token IDF"),
        ("Name semantics", "legal form equal / conflicting / one missing; acronym match (State Bank of India ↔ SBI); best DBA-variant alignment; skeleton Jaccard; digits in names agree/conflict; first/last token similarity"),
        ("Address", "same string metrics + alignment on the address core and on the full address; overlap coefficient, containment both ways; landmark cross-match"),
        ("Numbers", "number-set Jaccard, first number equal, conflict, shared count; postal-like equal / conflicting / one missing"),
        ("Cross-field", "name tokens found in the other record's address (both ways); joint name+address cosine and token-set ratios"),
        ("Rarity / chains", "how often the name occurs in the pool and in S1 (chain signal), address frequency (malls, shared buildings)"),
        ("Provenance", "each channel's score and rank, number of channels, meta-blocker score"),
        ("Record", "S3 flag, same-country flag, token counts, empty-address flags, candidates per S1, S1s per candidate"),
    ]) + "\n")
    m = rep.get("models", {})
    p1, p2 = m.get("pass1", {}), m.get("pass2", {})
    A(f"- **Pass-1** ({p1.get('backend')}): {p1.get('n_rows')} pairs × {p1.get('n_features')} features, "
      f"positive rate {f(p1.get('pos_rate'))}, early-stopped iterations {p1.get('iterations')}. Monotone constraints "
      "+1 on core similarities, −1 on conflict features; seeds " + f"{rep.get('seeds')}; test = mean of all fold models.")
    A("- **Pass-2 (collective):** pass-1 features + features computed from out-of-fold p1: rank within the S1 and "
      "gap to its best, relative p1, S1 max/second/sum, counts above 0.5/0.2; the S1's rank among all S1s claiming "
      "the same pool record, margin over the best competing S1, reciprocal-best flag; **sibling support** = max over "
      "the S1's other top candidates c′ of p1(s1,c′)·sim(c,c′)" +
      (" ; cross-encoder logit (xlm-roberta-base, 2-fold)" if rep.get("chosen_variant") == "with_ce" else "") +
      f". Variant used: `{rep.get('chosen_variant')}` (nested F0.5 by variant: {rep.get('variants_nested_f05')}).")
    A("- **Entity model:** h = P(at least one candidate is a true match) from per-S1 aggregates of p2 (top-1/2, gap, "
      "sum, entropy, counts above 0.3/0.5/0.7), best raw cosines, top-1 reciprocal flag and margin, S1 name rarity "
      "and length, address completeness.")
    A("- **Calibration and inference:** isotonic regression on out-of-fold p2 and h; test predictions are the "
      "mean of the fold models so the out-of-fold calibration applies.\n")
    imp1, imp2 = p1.get("importance_top30", {}), p2.get("importance_top30", {})
    rows = [(i + 1, a, f(imp1.get(a), 4), b, f(imp2.get(b), 4))
            for i, (a, b) in enumerate(zip(list(imp1)[:20], list(imp2)[:20]))]
    A("**Top features (normalized total gain)**\n")
    A(table(["#", "pass-1 feature", "gain", "pass-2 feature", "gain"], rows) + "\n")

    A("## 6. Decision layer\n")
    A("For one S1 with true set G and prediction P: **F0.5 = 1.25·TP / (|P| + 0.25·|G|)**; a true singleton scores "
      "1 only for an empty prediction. Hence (i) every entity is a one-point bet on \"is there any match?\", "
      "(ii) the first pick needs p > 0.5 but an extra pick needs p > F_current/1.25 (0.67 with 1 of 2 found), "
      "(iii) once matched, a false merge costs more than a miss.\n")
    A("- **Exclusivity:** for each pool record keep its best S1's probability and multiply the others by γ.")
    A("- **D1** threshold policy: top-1 if p ≥ t1; extras if p ≥ t2 and p ≥ r·p_top.")
    A("- **D2** exact expected F0.5 over top-k sets (Poisson-binomial DP; with the entity model, E[F(∅)] = 1 − h).")
    A("- **D3** = D2 after a logit recalibration p′ = σ(a·logit p + b) tuned on out-of-fold macro F0.5.")
    A("- **Selection:** nested CV over the full grid (γ, a, b, h on/off for D2/D3; t1, t2, r, γ for D1). "
      f"Chosen config: `{dec.get('best_config')}` → in-sample {f(dec.get('best_in_sample_f05'), 5)}, "
      f"nested {f(dec.get('nested_f05'), 5)}.\n")
    A("**Decoder ablation (same out-of-fold probabilities)**\n")
    A(table(["Decoder", "best config", "OOF macro F0.5"],
            [(k, str(v["config"]), v["f05_in_sample"]) for k, v in dec.get("decoder_ablation", {}).items()]) + "\n")

    A("## 7. Validation\n")
    A(f"Five folds over S1 entities stratified by country × match-count bucket (seed {cfg.get('seed')}); every "
      "learned stage (meta-blocker, pass-1, pass-2, entity model, decoder) reuses them. The metric is exact macro "
      "F0.5 over *all* S1s, including those with no candidates.\n")
    A(table(["Metric (train, out-of-fold)", "Value"], [
        ("macro F0.5, nested decoder tuning", oof.get("nested_f05")),
        ("macro F0.5, final policy", oof.get("macro_f05")),
        ("macro precision (non-empty predictions)", oof.get("macro_precision_nonempty")),
        ("macro recall (matched entities)", oof.get("macro_recall_matched")),
        ("singleton accuracy", oof.get("singleton_accuracy")),
        ("matched entities with a non-empty prediction", oof.get("matched_entity_nonempty_rate")),
    ]) + "\n")
    A(table(["Country", "n", "F0.5"], [(k, v["n"], v["f05"]) for k, v in oof.get("by_country", {}).items()]) + "\n")
    A(table(["True matches per S1", "n", "F0.5"], [(k, v["n"], v["f05"])
                                                   for k, v in oof.get("by_match_count", {}).items()]) + "\n")
    A("**Error budget** — lost macro-F0.5 points by cause (sums exactly to 1 − F0.5):\n")
    A(table(["Bucket", "Lost F0.5"], list(oof.get("error_budget", {}).items())) + "\n")
    if loco:
        A("**Leave-one-country-out** (full pipeline refitted without the held-out country — the proxy for France):\n")
        A(table(["Split", "held-out F0.5", "held-out PC", "train nested OOF", "half-pool F0.5"],
                [(k, v["heldout"]["macro_f05"], v["heldout_blocking"]["PC"], v["train_nested_oof_f05"],
                  v.get("half_pool_f05")) for k, v in loco.items()]) + "\n")
    else:
        A("Leave-one-country-out: run `python -m src.evaluate --data <dataset> --loco` to add the table here.\n")

    A("## 8. France and other unseen countries\n")
    A("""1. Nothing is keyed to {US, India}: country is only a partition key and a same-country flag.
2. Normalization is language-neutral and includes French street types, legal forms and elisions.
3. IDF and name/address frequencies are recomputed per test partition (label-free statistics of test inputs).
4. Features are scale-free (similarities, ranks, margins) with monotone constraints.
5. Every test S1 gets a row: the writer iterates the `test_source1` ids, never the predictions.
6. Test-time dashboard (below) compares each test country with the out-of-fold behaviour on train.
""")
    dash = rep.get("test_dashboard", {})
    rows = [(f"test {k}", v.get("n_s1"), v.get("no_candidates_rate"), v.get("pred_empty_rate"),
             v.get("mean_matches"), str(v.get("top1_prob_hist"))) for k, v in dash.get("test", {}).items()]
    rows += [(f"OOF {k}", v.get("n_s1"), v.get("no_candidates_rate"), v.get("pred_empty_rate"),
              v.get("mean_matches"), str(v.get("top1_prob_hist"))) for k, v in dash.get("oof_train", {}).items()]
    A(table(["Partition", "S1", "no candidates", "predicted empty", "mean matches", "top-1 p histogram (10 bins)"],
            rows) + "\n")

    A("## 9. Other relevant information\n")
    A("**Compliance.** No external data, APIs, geocoders, registries, address parsers (no libpostal) or scraping. "
      "Hand-written generic lexicons only (section 3). IDF/frequency statistics of the *test inputs* are computed "
      "label-free at inference time. No entity_id numbers, row order or country one-hot are used as features.\n")
    A(table(["Component", "Licence", "Parameters", "Used"], [
        ("LightGBM", "MIT", "–", "CPU backend"), ("XGBoost", "Apache-2.0", "–", "GPU backend"),
        ("scikit-learn", "BSD-3", "–", "TF-IDF, isotonic, folds"), ("rapidfuzz", "MIT", "–", "string metrics"),
        ("sparse_dot_topn", "Apache-2.0", "–", "sparse top-K"), ("numpy / scipy / pandas", "BSD", "–", "–"),
        ("FacebookAI/xlm-roberta-base", "MIT", "279M",
         "yes" if rep.get("chosen_variant") == "with_ce" else "no (not used in this run)"),
        ("intfloat/multilingual-e5-small", "MIT", "118M",
         "yes" if rep.get("neural_used", {}).get("dense_c8") else "no (not used in this run)"),
    ]) + "\n")
    A("Total model parameters used: well below the 8B cap.\n")
    A("**Reproducibility.**\n")
    A("```bash\ncd code/business_entity_resolution\npip install -r requirements.txt\n"
      "python -m src.run --data /path/to/student_resource/dataset --out ../../output\n"
      "python -m src.evaluate --data /path/to/student_resource/dataset --loco   # optional LOCO table\n```\n")
    A(table(["Item", "Value"], [("python", env.get("python")), ("platform", env.get("platform")),
                                ("CPU cores", env.get("cpu_count")), ("GPU", env.get("gpu") or "none"),
                                ("backend", rep.get("backend")), ("runtime (min)", f(rep.get("runtime_sec", 0) / 60, 1)),
                                ("peak RSS (GB)", rep.get("peak_rss_gb")),
                                *[(f"sha256 {k}", v) for k, v in rep.get("output_sha256", {}).items()],
                                ("stage timings (s)", str(rep.get("timings_sec")))]) + "\n")
    A(f"Validator: local mirror → {rep.get('local_check', ['–'])[:3]}; official → "
      f"{rep.get('official_validator', {}).get('pass', 'not found')}.\n")
    A("**Limitations and next steps.** The independence assumption of the expected-F decoder is corrected by the "
      "tuned recalibration and exclusivity, not modelled exactly; chains with identical names and addresses remain "
      "ambiguous; France is validated only through LOCO and the test dashboard (no French labels exist).\n")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(L), encoding="utf-8")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="reports")
    a = ap.parse_args()
    rd = Path(a.report)
    rep = json.load(open(rd / "run_report.json", encoding="utf-8"))
    lo = json.load(open(rd / "loco.json", encoding="utf-8")) if (rd / "loco.json").exists() else None
    print(write_documentation(rep, rd / "Documentation_template.md", loco=lo))
