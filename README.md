# Amazon ML Challenge 2026 — Business Entity Resolution

For every Source-1 business record, find all Source-2/Source-3 records of the same real-world business, across
~10M noisy records per split (US, India, and a test-only France). The score is macro F0.5 per Source-1 entity;
a singleton scores 1 only when nothing is predicted for it.

**Final public leaderboard: 0.978805 macro F0.5** (first submission 0.9496). Test-like holdout: 0.9828.

```
raw TSVs → learned normalization (native-script lexicon, website splitting, typo / padded-number repair)
  → 8 hashed TF-IDF blocking channels per country → LightGBM meta-blocker → top-25 per S1      = candidate_pairs.tsv
  → 103 pair features → LightGBM pass-1 (5-fold) → within-S1 collective features → LightGBM pass-2
  → 2 fine-tuned cross-encoders (multilingual-e5-small, xlm-roberta-base) on uncertain pairs
  → stacker fitted on a test-like holdout → isotonic calibration → exclusivity
  → exact expected-F0.5 decoding per S1                                                      = matching_results.tsv
```

The full write-up, with every number, is in [docs/methodology.md](docs/methodology.md).

## Results

| # | Version | Holdout F0.5 | Public LB |
|---|---|---|---|
| 1 | Blocking + LightGBM + expected-F0.5 decoding | 0.9662 | 0.949559 |
| 2 | + label-free test-shift correction | 0.9662 | 0.9540 |
| 3 | + 2× training S1s, larger LightGBM | 0.9683 | 0.9560 |
| 4 | + learned normalization, generator keys, 3 new blocking channels | 0.9769 | not submitted alone |
| 5 | + cross-encoder fusion (stacker), with test-shift correction | 0.9828 | 0.9780 – 0.978518 |
| 6 | **final: fusion without the test-shift correction** | 0.9828 | **0.978805** |

The holdout is carved out of train to look like test (whole cities, S1s not used in training, pool density matched to
test, scored by the exact test code path). Blocking recall of gold pairs on it rose from 95.8% (v1–3) to 97.7% (v4).

<details>
<summary>How each leaderboard variant of version 5/6 was produced</summary>

All from the same saved test probabilities and cross-encoder scores (`src.decide --big --reuse --stack --ce ce_outputs.zip --tag stack_ce …`):

| Variant | Flags | Public LB |
|---|---|---|
| A | `--variant shift` (pooled test-shift correction) | 0.9780 |
| A2 | `--variant shift --france-as india` | 0.978466 |
| B | `--variant shift --france-as us` | 0.978518 |
| **B2** | `--variant v3` (no shift correction) | **0.978805** |
| C | `--variant v3 --logit-shift us:0.25` | 0.978671 |

</details>

## Repository layout

```
.
├── run_final.sh          one-command reproduction, stages 0–7 (checkpointed, resumable)
├── requirements.txt      pinned environment (Python 3.11)
├── src/                  the pipeline, run as python -m src.<module>; all settings in src/config.yaml
├── kaggle_ce/            Kaggle GPU kit that fine-tunes the two cross-encoders (run.py + notebook)
├── docs/methodology.md   methodology write-up submitted with the solution
├── reports/              run reports of submissions 1–3 (first pipeline, larger LightGBM)
└── reports_v4/           run reports of the final pipeline (v4 normalization + blocking, fusion)
```

## Data

The challenge data is **not** in this repository (it may not be redistributed; get it from the challenge portal).
The scripts default to this workspace layout, and `DATA`, `CACHE` and `OUT` override it:

```
<workspace>/
├── student_resource/dataset/{train,test}/   challenge data
├── ber_cache/                               checkpoints (created, ~16 GB)
├── output/                                  matching_results.tsv, candidate_pairs.tsv (created)
└── code/business_entity_resolution/         this repository
```

```bash
git clone https://github.com/AnshBajpai05/Amazon-ML-challenge-2026.git code/business_entity_resolution
```

## Reproduce the final submission

```bash
cd code/business_entity_resolution
python3.11 -m venv .venv && source .venv/bin/activate        # Windows: Git Bash works too
pip install -r requirements.txt
DATA=/path/to/student_resource/dataset CE=/path/to/ce_outputs.zip bash run_final.sh   # -> ../../output/*.tsv
```

| Stage | Command (run by `run_final.sh`) | Time (16 cores, 16 GB RAM) |
|---|---|---|
| 0 | `python -m src.lexicon`: learned native-script → English word lexicon (from train gold pairs) + S1-name vocabulary | 2 min |
| 1 | `python -m src.run --set scale.train_only=true …`: normalize, blocking, meta-blocker, training pairs + features | 1.5 h |
| 2 | `python -m src.bigtrain train`: 400k training S1s, LightGBM pass-1 / pass-2 (5-fold) | 2.5 h |
| 3 | `python -m src.bigtrain proxy --n 60000`: test-like holdout from train, scored with the new models | 0.7 h |
| 4 | `python -m src.decide --big --fast --n 60000`: decision layer tuned by nested CV over cities | 5 min |
| 5 | `python -m src.bigtrain test`: every test S1 blocked and scored | 5.5 h |
| 6 | `python -m src.decide --big --reuse --variant shift`: LightGBM-only decoding, both output files, official validator | 10 min |
| 7 | `python -m src.decide --big --fast --stack --ce ce_outputs.zip --apply --variant v3`: LightGBM + cross-encoder stacker fitted on the holdout (city-grouped CV), decoding re-tuned, applied to test | 30 min |

Every stage is checkpointed in `CACHE`: rerun the same command to resume, or `bash run_final.sh N` to start at stage N.
A failed stage is retried up to 3 times. Reports (metrics, holdout error analysis, validator output) go to
`reports_final/`. Without `CE`, stage 6's LightGBM-only output is the final output.

**Cross-encoder scores (`CE=ce_outputs.zip`).** Two MIT-licensed multilingual transformers
(`intfloat/multilingual-e5-small`, `FacebookAI/xlm-roberta-base`) are fine-tuned as pair classifiers on a Kaggle GPU
(T4 ×2, ~4 h) with the kit in [kaggle_ce/](kaggle_ce/README.md) (checkpointed and resumable). The kit's data is
exported with `python -m src.ce_export`: hard training pairs of the training S1s (labels from train gold only) and the
uncertain (0.002 < p < 0.998) pairs of the holdout and of test. The notebook returns `ce_outputs.zip`, one logit per
pair and model. On the holdout's uncertain pairs, LightGBM has AUC 0.982, each transformer 0.991, and the fusion 0.994.

## Modules

```
src/run.py         entry point (first stage)      src/pipeline.py   stage orchestration + checkpoints
src/scale.py       partitioned train/test path    src/bigblock.py   per-country TF-IDF blocking channels
src/bigtrain.py    final model: train/proxy/test  src/proxy.py      test-like holdout from train
src/decide.py      decision layer, test decoding  src/decode.py     expected-F0.5 decoder
src/stack.py       LightGBM + cross-encoder stack src/lexicon.py    learned native-script lexicon (-> lexicon.json)
src/normalize.py   normalization                  src/blocking.py   channels + meta-blocking features
src/features.py    pair features                  src/collective.py pass-2 + entity features
src/models.py      GBDT OOF training              src/neural.py     optional in-pipeline cross-encoder / dense channel
src/ce_export.py   Kaggle kit data export         src/ce_export2.py phase-2 export (new candidates of v4)
src/metrics.py     F0.5, error budget, KPIs       src/eda.py        EDA gates G1-G10
src/cache.py       checkpoints                    src/track.py      progress tracking
src/evaluate.py    LOCO + scoring                 src/selftest.py   end-to-end self-test
src/docs.py        documentation writer           src/package.py    submission zip
src/compat.py      version/CPU shims              src/synth.py      synthetic test data
src/config.yaml    every K, tau, gamma, seed and grid
```

## Commands

| Command | What it does |
|---|---|
| `python -m src.selftest [--loco] [--keep DIR]` | ~1 min synthetic end-to-end test (crash/resume, determinism, validator); exit 0 = PASS |
| `python -m src.run --data D --out O [--team T --zip DIR]` | first-stage pipeline: train on `train/`, predict `test/`, validate, write reports + documentation |
| `  --cache DIR` / `--resume-from DIR` / `--no-cache` | checkpoint dir (default `O/../ber_cache`) / extra read-only checkpoint dirs / always recompute |
| `  --budget-hours H` | time budget; optional stages (cross-encoder, dense channel) are skipped if they would not fit |
| `  --set key.sub=value` | override any `src/config.yaml` value, e.g. `--set backend=lightgbm --set meta.cap=50` |
| `python -m src.decide --big --reuse [--stack --ce Z] [--variant v3\|shift] [--logit-shift cty:x] --out O` | re-decode saved test probabilities with another decision policy |
| `python -m src.evaluate --data D --loco [--density]` | leave-one-country-out (+ half-pool robustness check) → `reports/loco.json` |
| `python -m src.evaluate --score M --gold G [--cand C --test-dir T]` | score any prediction file against a gold file |
| `python -m src.docs --report reports` | regenerate `Documentation_template.md` from the reports |
| `python -m src.package --team T --out O --doc D` | build `<team>_submission.zip` in the required layout |
| `python -m src.synth --out DIR` | synthetic dataset with the challenge schema (tests only; never used for training) |

`--data` may be the `dataset` folder or any ancestor of it; it's searched for `train/train_source1.tsv`. The official
validator runs automatically when `utils/validate_submission.py` is found, and the SHA-256 of both outputs is recorded
in the run report.

## Reports

`output/`: `matching_results.tsv` (leaderboard file) and `candidate_pairs.tsv` (the meta-blocked set, exactly the set
the matcher scores). Every S1 id gets one row, and matches ⊆ candidates is asserted.

| File (in `reports*/`) | Content |
|---|---|
| `run.log` | full log: numbered stages with elapsed time and RSS, per-fold AUC/AP/logloss with ETA, summary block |
| `progress.json` | rewritten after every stage; on a crash it holds the failing stage, the traceback and all metrics so far |
| `run_report.json` | EDA gates, blocking KPIs per channel/country, model metrics per fold, calibration, decoder grid, OOF F0.5 per country / match count, error budget, timings, memory, environment, hashes |
| `bigtrain/manifest.json` | final model: training sample size, LightGBM params, features, per-pass training info, checkpoint keys |
| `decide*/decide_report.json` | holdout profile, nested-CV F0.5 of each decision step (by city), top policies, chosen policy, test-shift estimate, test summary |
| `feature_importance.csv`, `decoder_grid.csv` | feature gains (pass-1/pass-2), OOF F0.5 of every decoder configuration |

Large per-pair files (`oof_pairs.parquet`, `test_pairs.parquet`, …) are written too but not committed.

## Checkpoints and resuming

Every stage is checkpointed under a key that hashes its inputs: a data fingerprint, the config sections it uses,
the source of the modules it runs, and the upstream key. Every CV fold and each test chunk is checkpointed too.
Rerunning the same command after a crash or timeout resumes at the first unfinished stage, fold or chunk. Changing
only the decoder reuses everything before it; editing `features.py` recomputes the features and everything after.
Keys include the data paths as given, so pass `--data`/`--cache` the same way (relative or absolute) across stages.

## Hardware

* **CPU only** for the pipeline: LightGBM (deterministic, `force_row_wise`). Thread pools are sized by the
  cgroup-aware usable CPU count, not `os.cpu_count()`; override with `BER_THREADS`. Built and run on 16 cores / 16 GB RAM.
* **GPU** only for the cross-encoder kit (Kaggle T4 ×2). XGBoost `device=cuda` and the in-pipeline cross-encoder are
  optional first-stage boosters used only when a working CUDA device is found.
* Feature matrices are float32; blocking is chunked per partition with an exact sparse top-K.

## Compatibility (tested)

| Stack | Python | numpy | pandas | scikit-learn | lightgbm | xgboost | Result |
|---|---|---|---|---|---|---|---|
| pinned (`requirements.txt`) | 3.11 (Windows, 16 CPUs) | 2.3.3 | 2.3.3 | 1.7.2 | 4.6.0 | 3.2.0 | selftest + LOCO PASS |
| Kaggle-2025-like (Linux, 4-CPU quota) | 3.11 | 1.26.4 | 2.2.3 | 1.2.2 | 4.5.0 | 2.0.3 | selftest + LOCO PASS |
| latest (Linux, 4-CPU quota) | 3.12 | 2.5.3 | 3.0.6 | 1.9.1 | 4.7.0 | 3.4.1 | selftest + LOCO PASS |
| older, no pyarrow (Linux, 4-CPU quota) | 3.10 | 1.26.4 | 2.1.4 | 1.3.2 | 4.3.0 | 2.0.3 | selftest + LOCO PASS |

Results differ slightly between library versions (e.g. LightGBM 4.3 vs 4.7), but reruns within one environment are
byte-identical. rapidfuzz ≥ 3.0 works; sparse_dot_topn is optional (an exact SciPy fallback is used without it).

## Compliance

Only the provided data is used: no external data, APIs, geocoders, registries or address parsers. The hand-written
lexicons in `src/normalize.py` are generic linguistic knowledge (abbreviations, legal forms, landmark cues,
stopwords); `src/lexicon.json` is learned from the training gold pairs by `src/lexicon.py`. TF-IDF and frequency
statistics of the test inputs are computed label-free at inference time. entity_id numbers, row order and country
one-hots are never features. The only pretrained models are the two cross-encoders: `intfloat/multilingual-e5-small`
(MIT, 118M) and `FacebookAI/xlm-roberta-base` (MIT, 278M), both far under the 8B-parameter cap. Libraries: LightGBM
(MIT), XGBoost (Apache-2.0), scikit-learn (BSD-3), rapidfuzz (MIT), sparse_dot_topn (Apache-2.0), numpy/scipy/pandas
(BSD), transformers (Apache-2.0).
