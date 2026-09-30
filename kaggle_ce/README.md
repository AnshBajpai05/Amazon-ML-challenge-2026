# Cross-encoder ensemble for Business Entity Resolution (Kaggle kit)

Two transformer cross-encoders read both records' raw text (`name | address`, scripts and accents intact) and score
P(same business). They are fine-tuned on the **hard pairs of the training S1s** and score the **uncertain pairs of
the holdout and the test** (where LightGBM's decision can still change). The scores go back to the local pipeline,
which fuses them with LightGBM (stacking fitted on the holdout) before the expected-F0.5 decoder.

| Member | Model | License | Params | Default |
|---|---|---|---|---|
| `e5s` | `intfloat/multilingual-e5-small` | MIT | 118M | 3 epochs, bs 128, lr 5e-5 |
| `xlmr` | `FacebookAI/xlm-roberta-base` | MIT | 278M | 2 epochs, bs 64, lr 2e-5 |

## Build the kit
From the repository root, after the first-stage run (`python -m src.run` + `python -m src.proxy`):
```bash
python -m src.ce_export --out kaggle_ce/data      # writes the parquet files listed under Files
# zip kaggle_ce/ (run.py, ce_notebook.ipynb, requirements.txt, data/) as ber_ce_kit.zip
```

## Run it on Kaggle
1. **Datasets → New dataset** → upload `ber_ce_kit.zip` (Kaggle unpacks it). Name it e.g. `ber-ce-kit`.
2. **New notebook** → *File → Import notebook* → `ce_notebook.ipynb` from the kit (or copy its cells).
3. Settings: **Accelerator GPU T4 x2** (P100 works too, one model at a time), **Internet ON** (models download from
   the Hugging Face hub), add the `ber-ce-kit` dataset as input.
4. **Save Version → Save & Run All (Commit)**. The run continues without the browser; expected wall time about
   2-2.5 hours on T4 x2 (both models in parallel, one per GPU).
5. When the version finishes: **Output** tab → download `ce_outputs.zip` (scores, metrics and analysis, ~150-250 MB).
   Pass it to the local pipeline as `CE=/path/to/ce_outputs.zip bash run_final.sh 7`; stage 7 fuses the scores with
   LightGBM and writes the final submission file.

Interactive sessions work too: run all cells; the monitor cell can be interrupted and re-run at any time, the
training keeps running in the background.

## Checkpoints and resuming
* Training saves `ce_work/<tag>/ckpt/last.pt` (model, optimizer, scheduler, fp16 scaler, exact position in the
  epoch) every 1,000 steps; the best model by validation logloss is kept in `ce_work/<tag>/best/`.
* Scoring writes shards of 250k pairs; finished shards are never recomputed.
* Same session: just re-run the cells, every stage resumes.
* **New session after a timeout:** add the previous version's *output* as an input of the notebook; the config cell
  finds its `ce_work/*/ckpt/last.pt` automatically (`RESUME_FROM`) and training continues where it stopped.

## Files
| File | What |
|---|---|
| `run.py` | Train / score one cross-encoder (`python run.py all --model ... --tag ... --gpu 0`); `--smoke` = tiny CPU test |
| `ce_notebook.ipynb` | Environment check, data analysis, parallel training on 2 GPUs, live monitor, analysis artifacts, packaging |
| `requirements.txt` | Libraries (all preinstalled on Kaggle GPU images; torch deliberately not pinned) |
| `data/train_pairs.parquet` | Training pairs: `s1_id, cand_id, cty, y, p_gbdt, is_val` |
| `data/holdout_pairs.parquet` | Holdout pairs in the uncertain band: `s1_id, cand_id, cty, y, p_gbdt, fold` |
| `data/test_pairs.parquet` | Test pairs in the uncertain band: `s1_id, cand_id, cty, p_gbdt` |
| `data/records.parquet` | `id, name, addr` for every record referenced |
| `data/manifest.json` | Counts and export settings |

## Outputs (`ce_outputs.zip`)
`ce_work/<tag>/scores_{val,holdout,test}.parquet` (column `ce_<tag>` = logit), `metrics.json`, `train_log.csv`,
`config.json`, `run.log`, and `analysis/` (training curves, ROC/PR and calibration vs LightGBM, fusion lift on the
holdout, disagreement examples, EDA plots, `summary.json`).

## Troubleshooting
* *"no kernel image is available"*: the GPU (P100 = sm_60) is not supported by the image's torch build -> use T4.
* *Model download fails*: Internet is off -> turn it on, or attach the model as a Kaggle input and pass its folder
  as `model` in the config cell.
* *Out of GPU memory*: lower `bs` for that member in the config cell (checkpoints stay valid).

## Data in this kit
794,080 training pairs (195k training S1s, 37% true, 5% of S1s held out for validation), 807,809 holdout pairs
(28% true, city folds), 5,296,520 test pairs (France 0.81M, India 2.51M, US 1.98M) = the pairs whose LightGBM
probability lies in [0.002, 0.998], and 6.9M record texts. Every pair id is present in `records.parquet`.
