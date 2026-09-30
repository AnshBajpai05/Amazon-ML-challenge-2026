# Amazon ML Challenge 2026: Business Entity Resolution — Methodology

**Submission Date:** 27 September 2026  
**Final public leaderboard:** 0.978805 macro F0.5

---

## 1. Executive Summary
We resolve every Source-1 business against the ~10M Source-2/Source-3 records of its country with a
**blocking → gradient-boosted pair classifier → per-entity decision** pipeline, built and validated only on the
provided data. Three things carry the result:
1. **Data-driven normalization.** A native-script → English word lexicon is learned from the training gold
   pairs, and website/handle names are split with the vocabulary of Source-1 names.
2. **Multi-channel TF-IDF blocking** with a learned meta-blocker.
3. **A LightGBM + transformer ensemble** (two fine-tuned multilingual cross-encoders fused by a stacker), feeding
   an **F0.5-optimal decision layer**: expected-F0.5 decoding per entity, one owner per record, and a label-free
   correction for the harder test distribution, all tuned on a test-like holdout carved out of train.

---

## 2. Methodology

### 2.1 Problem Analysis
- **Scale.** Train has 2.21M Source-1 entities and 10.3M S2/S3 records (US, India). Test has 1.73M Source-1
  entities and 10.0M records (US, India, and the unseen **France**).
- **Structure found in train gold.**
  - 5.6% of S1 entities are singletons; the mean is 3.46 matches per S1.
  - Matches never cross countries.
  - A pool record belongs to at most one S1 (exclusivity is exact).
- **Noise patterns, quantified on gold pairs.** These are the pairs a word-level matcher misses:

| Pattern | Share of gold pairs | Example |
|---|---|---|
| Name written in an Indic script (9 scripts) | 8.9% | `Great Media Pvt Ltd` ↔ `ग्रेट मीडिया प्राइवेट लिमिटेड` |
| Website / handle form of the name | 5.9% | `Olaniq Pet Care` ↔ `olaniqpetcare.com`, `#Granddefense` |
| Zero-padded house number | 6.0% | `218 1st Street` ↔ `00218 1st St` |
| Candidate without an address | 4.3% | `Sterling PLLC` ↔ `Sterling PLLC Enterprises` (no address) |
| Digit-for-letter typos | 1.8% | `Hospitality` ↔ `Hospita1ity`, `Flint` ↔ `F1int` |

  Other patterns: abbreviations (Pvt/Private, St/Street), legal-suffix changes, word transpositions, DBA names,
  state names vs codes, landmark-based and partial addresses.
- **Look-alike records.** "Sibling" businesses such as `X Groupe` / `X Distribution` / `X & Fils` sit at nearby
  house numbers. At equal pool density, **test has about 2× train's mid-probability (0.5–0.9) candidates per
  S1**, i.e. more look-alikes. This was confirmed on the leaderboard (see 5).
- **No leakage.** Record order and ID numbers carry no signal (correlation ≈ 0.00).

### 2.2 Solution Strategy
**Approach Type:** Blocking + Classifier (GBDT) + collective features + optimal per-entity decoding (hybrid)  
**Core Innovation:**
- **Learned normalization:** native-script lexicon from the training pairs, website splitting, typo and number
  repair.
- **Test-like holdout:** built from train. Whole cities, S1s removed until pool density matches test, fold-mean
  models. It stands in for the leaderboard, so every decision-layer change is measured out-of-sample.
- **Label-free test-shift correction** of calibrated probabilities.

```
raw TSVs → normalize (per record) → per-country partitions
  → 7 blocking channels (hashed TF-IDF, sparse top-K) → union → meta-blocker (LightGBM) → top-25 per S1  = candidate_pairs.tsv
  → 103 pair features → LightGBM pass-1 (5-fold OOF) → 12 within-S1 collective features → LightGBM pass-2
  → + 2 cross-encoders (multilingual-e5-small, xlm-roberta-base) on uncertain pairs → stacker (fitted on holdout)
  → isotonic calibration → test-shift correction → exclusivity → expected-F0.5 decoding per S1       = matching_results.tsv
```

---

## 3. Candidate Generation (Blocking)
- **Partitioning:** by country (an open set of labels; France needs no special handling).
- **Representation:** hashed bag-of-tokens (2²² features), sublinear TF, IDF computed per partition without
  labels.
  - Name tokens are words, phonetic skeletons, 4-letter prefixes and word bigrams.
  - Very common tokens are pruned (posting list > 2% of the partition or > 3,000 records).
- **Blocking keys (channels)**, exact sparse top-K by cosine:

| Channel | Query | K |
|---|---|---|
| c2 name | S1 → pool | 30 |
| c4 address | S1 → pool | 20 |
| c5 name + address | S1 → pool | 50 |
| c6 reverse name + address | pool → S1 | 10 |
| c9 reverse name | pool → S1 | 10 |
| **c1 generator keys** (concatenated / sorted name, initials forms: `olaniqpetcare`, `sg`, `ccfields`) | S1 → pool | 10 |
| **c7 house number × locality word** (`28#chennai`) | S1 → pool | 12 |
| **c8 reverse name, address-less pool records only** | pool → S1 | 25 |

- **Meta-blocker:** a small LightGBM on 27 blocking features (channel scores and ranks, exact cosines, postal
  agreement), trained out-of-fold on train. It keeps the **top 25 candidates per S1**.
- **Candidate pairs generated (test):** ≈43.3M (25 per S1, from a blocking union of ~135 per S1).
- **How true matches were kept:**
  - Channels are complementary (names, addresses, both, and reverse competition).
  - The normalization fixes above make scripts, websites, typos and padded numbers comparable before blocking.
  - The two new channels target the two largest miss sources (Indic/website names through the address key, and
    address-less records).
  - Holdout recall of gold pairs: union **98.4%** (previously 96.6%), after the top-25 cut **97.7%** (previously
    95.8%). On the training sample: 98.6% / 98.0% (India 97.2%, US 98.4% after the cut).

---

## 4. Matching Model

**Features used** (103 pair features plus 12 collective features):
- **Name:** Jaro-Winkler, Levenshtein, ratio, token-sort and token-set ratios, partial ratio, LCS, prefix
  match, first/last-token similarity. Also IDF-weighted soft token alignment (coverage both ways, shared/unshared
  rarity), phonetic-skeleton Jaccard, acronym match, legal-form agreement/conflict, and numbers in names.
- **Address:** the same string measures on normalized addresses, soft alignment, overlap/containment, landmark
  cross-match, house-number Jaccard/first-number/conflict, postal agreement/conflict/missing, and name contained
  in address.
- **Other:**
  - token counts, empty-address flags, name/address frequency in the pool and S1 (rarity), S2 vs S3 source;
  - blocking provenance: channel scores and ranks, number of channels, the meta-blocker score;
  - **within-S1 collective (pass-2):** pass-1 probability, its rank within the S1, gap to the S1's best,
    relative probability, and count of strong/medium candidates. Also sibling support: how strongly the S1's
    other likely matches resemble this record.

**Model type:** LightGBM (MIT)
- Pass-1 and pass-2: 127 leaves, learning rate 0.05, ~3,500–3,900 trees per fold, early stopping, monotone
  constraints on similarity features.
- 5-fold cross-validation grouped by S1, stratified by country × number of matches.
- Trained on **400k S1 entities** (~10M candidate pairs), with the negative mask: all positives, each S1's
  top-12 candidates by meta score, and 35% of the rest.
- Test predictions are the mean of the 5 fold models.
- No external data or services are used, and no model is larger than a few MB, far below the 8B-parameter
  limit.

**Transformer cross-encoders (ensemble members):**
- `intfloat/multilingual-e5-small` (MIT, 118M) and `FacebookAI/xlm-roberta-base` (MIT, 278M), fine-tuned as pair
  classifiers on `name | address` of both records (raw text, scripts intact). Kaggle T4 ×2, 3 and 2 epochs, fp16.
- Training data: 794k hard pairs of the training S1s (labels from train gold). They score the uncertain
  pairs (LightGBM p in [0.002, 0.998]) of the holdout and test.
- On the holdout's uncertain pairs: LightGBM AUC 0.982, each transformer 0.991, **fusion 0.994** (log-loss
  0.138 → 0.076).
- **Stacker:** a LightGBM on logit p2, logit p1, meta score, both transformer logits (with a has-score flag),
  group context and competition for the record. Fitted on the holdout with city folds; its out-of-fold
  probabilities feed the decoder tuning.

**Threshold selection method:** there is no global threshold. Per S1, we choose the set size k that
maximizes **expected F0.5** over its top-k candidates (exact Poisson-binomial dynamic program, including the
empty set for singletons), on calibrated probabilities:
1. **Isotonic calibration**, fitted on a **test-like holdout from train**:
   - whole cities of train S1s not used in training;
   - extra S1s removed until pool-per-S1 matches test (5.8);
   - scored exactly like test.
2. **Label-free test-shift correction per country** (used up to version 5; after the transformer fusion it no longer
   helped on the leaderboard, so the final submission omits it). Test's extra look-alike mass per probability band is
   treated as negatives, and the proxy's true pairs per S1 per band are kept. This only lowers probabilities.
3. **Exclusivity:** a pool record goes to its strongest claim only.
4. **Logit recalibration (a, b)**, tuned with nested cross-validation over cities.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):**
  - Test-like holdout (126k S1s, nested cross-validation by city): **0.9828**.
  - Earlier versions on the holdout: 0.9662 → 0.9683 → 0.9769 (learned normalization) → **0.9828** (+ transformer
    fusion).
  - Public leaderboard progression: 0.9496 (first model) → 0.9540 (test-shift correction) → 0.9560 (larger
    model) → 0.9785 (learned normalization + transformer fusion, with test-shift correction) → **0.9788** (final:
    fusion without the test-shift correction).
- **Common false positives (wrong merges):**
  - **Look-alike sibling businesses:** the same brand at a neighbouring house number, or a different brand at
    the same address.
  - **Generic names** in dense cities.
  - False merges on singletons (holdout singleton accuracy 96.5%).
  - 96% of false positives are records that belong to no other S1, so exclusivity alone cannot remove them.
    This is why the test-shift correction matters.
- **Common false negatives (missed matches):**
  - **Before the final version:** native-script names, website names, zero-padded numbers and digit typos. They
    were missed by blocking and under-scored by the model (73% of blocking misses) and are fixed by the learned
    normalization.
  - **Remaining:** address-less records with generic names, and heavily abbreviated names (e.g. `sg.com` for
    `Stewart, Gustafson & Cooley`).
  - Extra matches in large groups whose text shares little with the S1.

---

## 6. Conclusion
A fully offline pipeline, with blocking, a GBDT, two fine-tuned transformer cross-encoders and an optimal
per-entity decoder, reaches **0.9788** on the public leaderboard (0.9496 for our first model).
- **What mattered:** reading the noise correctly, above all Indic scripts, website names, typos and padded
  numbers. We learned these from the training data itself.
- **Honest validation:** a test-like holdout, used instead of a random out-of-fold split, which had
  overestimated our score by 0.017.
- **Model size mattered less than data coverage:** the larger model added only +0.002, the learned
  normalization +0.009 and the transformer fusion +0.006 on the holdout.

---

## Appendix

### A. Code Artefacts
This repository (submitted as `code/business_entity_resolution/`; Python 3.11, `requirements.txt` pins versions).
Reproduce both output files with one command:
```
DATA=/path/to/student_resource/dataset CE=/path/to/ce_outputs.zip bash run_final.sh   # stages 0-7, checkpointed and resumable
```
| Stage | Entry point | Output |
|---|---|---|
| 0 | `python -m src.lexicon` | `src/lexicon.json` (learned native-script lexicon + S1-name vocabulary) |
| 1 | `python -m src.run --set scale.train_only=true` | normalized data, blocking union, meta-blocker, training pairs + features |
| 2 | `python -m src.bigtrain train` | LightGBM pass-1 / pass-2 fold models |
| 3 | `python -m src.bigtrain proxy` | test-like holdout scored with the models |
| 4 | `python -m src.decide --big --fast` | decision layer tuned by nested cross-validation (report `reports_final/decide_big/`) |
| 5 | `python -m src.bigtrain test` | probabilities for every test candidate pair |
| 6 | `python -m src.decide --big --reuse --variant shift` | LightGBM-only `output/matching_results.tsv`, `output/candidate_pairs.tsv`, official validator |
| 7 | `python -m src.decide --big --fast --stack --ce ce_outputs.zip --apply --variant v3` | final (fused) `output/matching_results.tsv` |

Modules:
- `normalize.py` (normalization);
- `bigblock.py`, `blocking.py` (blocking channels, meta features);
- `features.py`, `collective.py` (pair and collective features);
- `models.py` (cross-validated GBDT);
- `scale.py` (partitioned train/test orchestration);
- `bigtrain.py` (final model);
- `proxy.py` (test-like holdout);
- `decide.py`, `decode.py` (decision layer);
- `stack.py` (LightGBM + cross-encoder stacker), `lexicon.py` (learned lexicon), `ce_export.py` (Kaggle kit data);
- `cache.py`, `track.py` (checkpoints, progress).

### B. Additional Results
| Version | Change | Holdout F0.5 | Public LB |
|---|---|---|---|
| 1 | Blocking + LightGBM + expected-F0.5 decoding | 0.9662 | 0.9496 |
| 2 | + label-free test-shift correction | 0.9662 | 0.9540 |
| 3 | + 2× training S1s, larger LightGBM | 0.9683 | 0.9560 |
| 4 | + learned normalization, generator keys, 3 new blocking channels | 0.9769 | (not submitted alone) |
| 5 | + transformer cross-encoder fusion (stacker), with test-shift correction | 0.9828 | 0.9785 |
| 6 | **final:** fusion without the test-shift correction (the fused model resolves test's harder records itself) | 0.9828 | **0.9788** |

Blocking recall on the holdout (share of gold pairs among candidates): v1–3 95.8%, v4 97.7%.
