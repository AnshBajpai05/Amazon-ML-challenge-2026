# Business Entity Resolution — Methodology

**Team:** team  
*Generated automatically from `reports/run_report.json` of the run that produced `output/`.*  
Run started 2026-09-26 16:09:29, runtime 164.0 min, peak RSS 10.18 GB, backend `lightgbm`, GPU: none.

**Headline (out-of-fold on train, nested CV — decoder tuned on 4 folds, scored on the 5th):** macro F0.5 = **0.96729**. Public leaderboard: _fill in after upload_.

## 1. Methodology overview

```
train/ & test/ TSVs -> safe read (sep=\t, dtype=str, keep_default_na=False, QUOTE_NONE, UTF-8)
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
```

**Seven design decisions**

1. **Decide per entity, optimizing the metric itself.** F0.5 is computed per S1, so each S1's output set
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
7. **One command reproduces everything.** Pinned environment, fixed seeds, SHA-256 of both outputs.

## 2. Data understanding (EDA gates)

**G1 — records per source × country**

| source | country | rows |
|---|---|

| Gate | Result | Decision it set |
|---|---|---|
| G2 matches per S1 (0 / 1 / 2 / 3+) | – / – / – / – | singleton gate weight; K per channel ≥ 3× p99 match count |
| G2 p99 / max matches per S1 | None | decoder keeps top-15 per S1 |
| G2 source share of matches | None | source flag feature |
| G3 pool ids in ≥ 2 gold lists | None | exclusivity enabled in the decoder grid (γ tuned) |
| G4 share of pool matched to some S1 | – | precision focus; reverse kNN |
| G5 matched pairs with same country | – | partition mode = `country` |
| G6 matched S1 with ≥ 2 matches from one source | – | 2-hop blocking + sibling support |
| G8 S1 records sharing a normalized name (chains) | None | name-frequency features; address decides chains |

**G10 — placeholder values ('', NA, N/A, -, None, …)** — read with `keep_default_na=False`, so a business literally named "NA" stays a string:

| split | source | column | count |
|---|---|

**G9 — drift per country (mean lengths)**

| split | source | country | n | name len | addr len | addr empty |
|---|---|---|---|---|

## 3. Normalization (country-agnostic, standard library + rapidfuzz only)

- **Unicode folding:** NFKC → lowercase → ligatures (œ→oe, æ→ae, ß→ss) → strip accents. No Unidecode (GPL).
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

## 4. Candidate generation / blocking

Partition key: normalized country label (mode `country`; train same-country match rate –). TF-IDF vocabularies and IDF are fitted per partition on S1 ∪ pool of the split being searched (no labels), so France gets its own weights. Exact sparse top-K (`sparse_dot_topn`, verified identical to brute force), never an approximate index.

| # | Channel | Representation | K |
|---|---|---|---|
| c1 | Name, character | char_wb 3–4-gram TF-IDF, sublinear, max_df 0.2 | 30 |
| c2 | Name, word | canonical core tokens, TF-IDF | 30 |
| c3 | Name, skeleton | skeleton tokens, TF-IDF | 15 |
| c4 | Address, character | 3–4-gram TF-IDF on the address core | 20 |
| c5 | Name + address | character n-grams of both | 30 |
| c6 | Reverse kNN | each pool record → its top S1s on c5 | 5 |
| c7 | 2-hop | pool self-kNN (cos ≥ 0.6) of each S1's top-3 c5 hits | 15 |
| c8 | Dense (optional) | multilingual-e5-small cosine | 20 (off) |
| c9 | Reverse name kNN | each pool record → its top S1s on the name char TF-IDF | 10 |

S1 records without an address can only be found through their name, so they search c1–c3 2.0× deeper; c9 does the same from the pool side (an address-less pool record inside a crowded name neighbourhood, e.g. chains).

**Learned meta-blocking** (the last filter, so its output *is* `candidate_pairs.tsv`): a small GBDT on channel scores/ranks, channel count and exact name/address/joint cosines + postal agreement, trained out-of-fold on the shared folds. Keep a pair if it is in its S1's top-3 by meta score or meta ≥ τ = 0.0000, capped at 25 per S1; τ keeps ≥ 0.995 of the union's true pairs out-of-fold.

**Blocking KPIs (train, labelled)**

| Stage | PC (recall ceiling) | Entity completeness | RR | PQ | cand/S1 mean | p95 | max |
|---|---|---|---|---|---|---|---|
| union of channels | 0.9673 | 0.9029 | 1.0000 | 0.0427 | 78.3918 | 144.0000 | 2136 |
| after meta-blocking (= candidate file) | 0.9603 | 0.8842 | 1.0000 | 0.1329 | 24.9996 | 25.0000 | 25 |

| Country | PC final | Entity completeness | cand/S1 |
|---|---|---|---|
| india | 0.9395 | 0.8353 | 24.9996 |
| us | 0.9742 | 0.9167 | 24.9996 |

**Per-channel contribution (train union):** true pairs found, and true pairs found by that channel only.

| Channel | true pairs found | unique true pairs |
|---|---|---|
| c1 | 0 | 0 |
| c2 | 382611 | 486 |
| c3 | 0 | 0 |
| c4 | 574455 | 816 |
| c5 | 655980 | 3853 |
| c6 | 658505 | 6949 |
| c7 | 0 | 0 |
| c8 | 0 | 0 |
| c9 | 393808 | 755 |

Test: None union pairs → 43313208 candidates (25.00 per S1, RR 0.999997).

## 5. Model architecture and feature engineering

| Family | Features |
|---|---|
| Name strings | Jaro-Winkler, normalized Levenshtein, ratio, token-sort/set, partial, LCS, prefix, on core / full / raw names; char, word and skeleton TF-IDF cosines |
| Soft alignment (key) | IDF-weighted greedy one-to-one token alignment with abbreviation/typo/skeleton-aware token similarity: soft Dice, coverage both ways, **max IDF of an unmatched token on each side**, shared-token IDF |
| Name semantics | legal form equal / conflicting / one missing; acronym match (State Bank of India ↔ SBI); best DBA-variant alignment; skeleton Jaccard; digits in names agree/conflict; first/last token similarity |
| Address | same string metrics + alignment on the address core and on the full address; overlap coefficient, containment both ways; landmark cross-match |
| Numbers | number-set Jaccard, first number equal, conflict, shared count; postal-like equal / conflicting / one missing |
| Cross-field | name tokens found in the other record's address (both ways); joint name+address cosine and token-set ratios |
| Rarity / chains | how often the name occurs in the pool and in S1 (chain signal), address frequency (malls, shared buildings) |
| Provenance | each channel's score and rank, number of channels, meta-blocker score |
| Record | S3 flag, same-country flag, token counts, empty-address flags, candidates per S1, S1s per candidate |

- **Pass-1** (lightgbm): 4999923 pairs × 103 features, positive rate 0.1329, early-stopped iterations [1529, 1824, 1760, 1716, 2004]. Monotone constraints +1 on core similarities, −1 on conflict features; seeds [42]; test = mean of all fold models.
- **Pass-2 (collective):** pass-1 features + features computed from out-of-fold p1: rank within the S1 and gap to its best, relative p1, S1 max/second/sum, counts above 0.5/0.2; the S1's rank among all S1s claiming the same pool record, margin over the best competing S1, reciprocal-best flag; **sibling support** = max over the S1's other top candidates c′ of p1(s1,c′)·sim(c,c′). Variant used: `pass2` (nested F0.5 by variant: {'pass1': 0.9657022206301711, 'pass2': 0.967288683267983}).
- **Entity model:** h = P(at least one candidate is a true match) from per-S1 aggregates of p2 (top-1/2, gap, sum, entropy, counts above 0.3/0.5/0.7), best raw cosines, top-1 reciprocal flag and margin, S1 name rarity and length, address completeness.
- **Calibration and inference:** isotonic regression on out-of-fold p2 and h; test predictions are the mean of the fold models so the out-of-fold calibration applies.

**Top features (normalized total gain)**

| # | pass-1 feature | gain | pass-2 feature | gain |
|---|---|---|---|---|
| 1 | meta | 0.6322 | p1_rel | 0.4962 |
| 2 | num_jacc | 0.0616 | p1 | 0.4557 |
| 3 | k_c6 | 0.0520 | meta | 0.0151 |
| 4 | j_tset | 0.0355 | gap_to_s1_best | 0.0105 |
| 5 | nf_jw | 0.0168 | s1_p1_sum | 0.0071 |
| 6 | j_tsort | 0.0147 | s1_p1_max | 0.0052 |
| 7 | a_freq_b | 0.0099 | num_jacc | 0.0009 |
| 8 | n_ratio | 0.0095 | s1_p1_2nd | 0.0008 |
| 9 | n_partial | 0.0093 | sib_support | 0.0007 |
| 10 | num_conf | 0.0081 | nf_jw | 0.0005 |
| 11 | n_mu_b | 0.0081 | j_tsort | 0.0004 |
| 12 | nf_tset | 0.0066 | a_ovl | 0.0004 |
| 13 | nr_tset | 0.0060 | rank_in_s1 | 0.0004 |
| 14 | num_first_eq | 0.0049 | sib_name_sim | 0.0003 |
| 15 | n_last_sim | 0.0048 | n_strong_in_s1 | 0.0003 |
| 16 | a_cov_b | 0.0046 | n_partial | 0.0003 |
| 17 | ntok_n_b | 0.0044 | nr_tset | 0.0002 |
| 18 | legal_conf | 0.0043 | n_freq_s1_a | 0.0002 |
| 19 | n_freq_s1_b | 0.0043 | k_c5 | 0.0002 |
| 20 | s_c6 | 0.0042 | nf_tset | 0.0002 |

## 6. Decision layer

For one S1 with true set G and prediction P: **F0.5 = 1.25·TP / (|P| + 0.25·|G|)**; a true singleton scores 1 only for an empty prediction. Hence (i) every entity is a one-point bet on "is there any match?", (ii) the first pick needs p > 0.5 but an extra pick needs p > F_current/1.25 (0.67 with 1 of 2 found), (iii) once matched, a false merge costs more than a miss.

- **Exclusivity:** for each pool record keep its best S1's probability and multiply the others by γ.
- **D1** threshold policy: top-1 if p ≥ t1; extras if p ≥ t2 and p ≥ r·p_top.
- **D2** exact expected F0.5 over top-k sets (Poisson-binomial DP; with the entity model, E[F(∅)] = 1 − h).
- **D3** = D2 after a logit recalibration p′ = σ(a·logit p + b) tuned on out-of-fold macro F0.5.
- **Selection:** nested CV over the full grid (γ, a, b, h on/off for D2/D3; t1, t2, r, γ for D1). Chosen config: `['d3', 0.0, 1.5, -0.25, False]` → in-sample 0.96739, nested 0.96729.

**Decoder ablation (same out-of-fold probabilities)**

| Decoder | best config | OOF macro F0.5 |
|---|---|---|
| global_threshold (d1, r=0, t1=t2, no exclusivity) | ['d1', 1.0, 0.7, 0.7, 0.0] | 0.9668 |
| d1 threshold policy | ['d1', 0.0, 0.5, 0.5, 0.7] | 0.9673 |
| d2 expected-F0.5 (a=1,b=0, no h, no exclusivity) | ['d3', 1.0, 1.0, 0.0, False] | 0.9673 |
| d2 + entity model h | None | – |
| d3 recalibrated + h + exclusivity (full grid) | ['d3', 0.0, 1.5, -0.25, False] | 0.9674 |

## 7. Validation

Five folds over S1 entities stratified by country × match-count bucket (seed 42); every learned stage (meta-blocker, pass-1, pass-2, entity model, decoder) reuses them. The metric is exact macro F0.5 over *all* S1s, including those with no candidates.

| Metric (train, out-of-fold) | Value |
|---|---|
| macro F0.5, nested decoder tuning | 0.9673 |
| macro F0.5, final policy | 0.9674 |
| macro precision (non-empty predictions) | 0.9922 |
| macro recall (matched entities) | 0.9251 |
| singleton accuracy | 0.9652 |
| matched entities with a non-empty prediction | 0.9920 |

| Country | n | F0.5 |
|---|---|---|
| india | 79800 | 0.9567 |
| us | 120200 | 0.9745 |

| True matches per S1 | n | F0.5 |
|---|---|---|
| 0 | 11240 | 0.9652 |
| 1 | 10772 | 0.9033 |
| 2 | 33882 | 0.9599 |
| 3+ | 144106 | 0.9741 |

**Error budget** — lost macro-F0.5 points by cause (sums exactly to 1 − F0.5):

| Bucket | Lost F0.5 |
|---|---|
| missed_extra_matches | 0.0188 |
| extra_false_positives | 0.0042 |
| missed_entity_blocking | 0.0041 |
| missed_entity_model | 0.0034 |
| singleton_false_merge | 0.0020 |
| wrong_pick_only | 0.0002 |

Leave-one-country-out: run `python -m src.evaluate --data <dataset> --loco` to add the table here.

## 8. France and other unseen countries

1. Nothing is keyed to {US, India}: country is only a partition key and a same-country flag.
2. Normalization is language-neutral and includes French street types, legal forms and elisions.
3. IDF and name/address frequencies are recomputed per test partition (label-free statistics of test inputs).
4. Features are scale-free (similarities, ranks, margins) with monotone constraints.
5. Every test S1 gets a row: the writer iterates the `test_source1` ids, never the predictions.
6. Test-time dashboard (below) compares each test country with the out-of-fold behaviour on train.

| Partition | S1 | no candidates | predicted empty | mean matches | top-1 p histogram (10 bins) |
|---|---|---|---|---|---|
| test france | 259452 | 0.0000 | 0.0579 | 3.2179 | [11795, 1343, 813, 584, 478, 382, 542, 599, 524, 242392] |
| test india | 809986 | 0.0000 | 0.0653 | 3.1848 | [45275, 3025, 2021, 1361, 1104, 944, 1607, 1938, 2041, 750670] |
| test us | 663106 | 0.0000 | 0.0571 | 3.3974 | [30851, 3013, 1776, 1264, 973, 644, 987, 1092, 1057, 621449] |

## 9. Other relevant information

**Compliance.** No external data, APIs, geocoders, registries, address parsers (no libpostal) or scraping. Hand-written generic lexicons only (section 3). IDF/frequency statistics of the *test inputs* are computed label-free at inference time. No entity_id numbers, row order or country one-hot are used as features.

| Component | Licence | Parameters | Used |
|---|---|---|---|
| LightGBM | MIT | – | CPU backend |
| XGBoost | Apache-2.0 | – | GPU backend |
| scikit-learn | BSD-3 | – | TF-IDF, isotonic, folds |
| rapidfuzz | MIT | – | string metrics |
| sparse_dot_topn | Apache-2.0 | – | sparse top-K |
| numpy / scipy / pandas | BSD | – | – |
| FacebookAI/xlm-roberta-base | MIT | 279M | no (not used in this run) |
| intfloat/multilingual-e5-small | MIT | 118M | no (not used in this run) |

Total model parameters used: well below the 8B cap.

**Reproducibility.**

```bash
cd code/business_entity_resolution
pip install -r requirements.txt
python -m src.run --data /path/to/student_resource/dataset --out ../../output
python -m src.evaluate --data /path/to/student_resource/dataset --loco   # optional LOCO table
```

| Item | Value |
|---|---|
| python | 3.11.8 |
| platform | Windows-10-10.0.26200-SP0 |
| CPU cores | 16 |
| GPU | none |
| backend | lightgbm |
| runtime (min) | 164.0 |
| peak RSS (GB) | 10.1800 |
| sha256 matching_results.tsv | 45ba4b3fda1277a2070f458fcfdb5021db53b78e1600d24bd1f898252c5f8a39 |
| sha256 candidate_pairs.tsv | bd7b418257d955fb7ac6d3253d4e451ef2647c48f17b958e7dc50bedd13c1905 |
| stage timings (s) | {'load_data': 14.8, 'normalize_sources': 0.0, 'train_sample': 9.5, 'meta_blocker': 46.0, 'train_features': 10.2, 'pass1': 6.6, 'pass2': 13.4, 'decoder_tuning': 371.1, 'test[france]': 2.7, 'test[india]': 5170.3, 'test[us]': 3873.3, 'decode_test': 187.1, 'write_and_validate': 111.2} |

Validator: local mirror → ['PASS']; official → True.

**Limitations and next steps.** The independence assumption of the expected-F decoder is corrected by the tuned recalibration and exclusivity, not modelled exactly; chains with identical names and addresses remain ambiguous; France is validated only through LOCO and the test dashboard (no French labels exist).
