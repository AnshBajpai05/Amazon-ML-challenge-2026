#!/usr/bin/env bash
# Reproduces output/matching_results.tsv and output/candidate_pairs.tsv from the raw data (the submitted pipeline).
#   DATA=/path/to/student_resource/dataset bash run_final.sh [first_stage 0..6]
# Every stage is checkpointed in $CACHE: re-running resumes; a failed stage is retried up to 3 times.
#   0 lexicon      learned native-script word lexicon + S1 name vocabulary (train data only)   ~2 min
#   1 base         normalize, blocking, meta-blocker, training pairs + features (train_only)   ~1.5 h
#   2 train        2x training S1s, LightGBM pass-1 / pass-2                                    ~2.5 h
#   3 holdout      test-like holdout from train, scored with the new models                    ~0.7 h
#   4 tune         decision layer tuned on the holdout (nested CV by city)                      ~5 min
#   5 test         every test S1 scored                                                         ~5.5 h
#   6 decide       shift-corrected expected-F0.5 decoding -> matching_results.tsv + candidates  ~10 min
#   7 fuse         (when CE=ce_outputs.zip is given) LightGBM + cross-encoder stacker fitted on the holdout,
#                  policy re-tuned by nested CV, applied to test -> final matching_results.tsv   ~30 min
#                  ce_outputs.zip comes from the Kaggle GPU kit in kaggle_ce/ (see kaggle_ce/README.md)
cd "$(dirname "$0")" || exit 1
export PYTHONIOENCODING=utf-8          # native-script text in logs
DATA=${DATA:-../../student_resource/dataset}
CACHE=${CACHE:-../../ber_cache}
OUT=${OUT:-../../output}
R=${REPORT:-reports_final}
LOG=${LOG:-run_final.log}
CE=${CE:-}                 # optional: ce_outputs.zip from the Kaggle cross-encoder kit
run() {
  for try in 1 2 3; do
    echo "$(date '+%F %T') >> stage $1 (try $try): ${*:2}" | tee -a "$LOG"
    if python -m "${@:2}" >> "$LOG" 2>&1; then
      echo "$(date '+%F %T') << stage $1 OK" | tee -a "$LOG"; return 0
    fi
    echo "$(date '+%F %T') !! stage $1 failed (try $try)" | tee -a "$LOG"; sleep 20
  done
  echo "$(date '+%F %T') XX stage $1 gave up" | tee -a "$LOG"; exit 1
}
FIRST=${1:-0}
[ "$FIRST" -le 0 ] && run 0 src.lexicon --data "$DATA"
[ "$FIRST" -le 1 ] && run 1 src.run --data "$DATA" --report "$R" --cache "$CACHE" --set scale.train_only=true \
    --set meta.cap=25 --set meta.recall_ratio=0.995 --set model.lgb.learning_rate=0.1 --set "decoder.gammas=[0.0,1.0]"
[ "$FIRST" -le 2 ] && run 2 src.bigtrain train --data "$DATA" --cache "$CACHE" --report "$R"
[ "$FIRST" -le 3 ] && run 3 src.bigtrain proxy --data "$DATA" --cache "$CACHE" --report "$R" --n 60000
[ "$FIRST" -le 4 ] && run 4 src.decide --big --fast --n 60000 --data "$DATA" --cache "$CACHE" --report "$R"
[ "$FIRST" -le 5 ] && run 5 src.bigtrain test --data "$DATA" --cache "$CACHE" --report "$R"
[ "$FIRST" -le 6 ] && run 6 src.decide --big --reuse --variant shift --n 60000 --data "$DATA" --cache "$CACHE" \
    --report "$R" --out "$OUT"
[ "$FIRST" -le 7 ] && [ -n "$CE" ] && run 7 src.decide --big --fast --stack --ce "$CE" --tag stack_ce --apply \
    --variant v3 --n 60000 --data "$DATA" --cache "$CACHE" --report "$R" --out "$OUT"
echo "$(date '+%F %T') ALL DONE -> $OUT/matching_results.tsv, $OUT/candidate_pairs.tsv" | tee -a "$LOG"
