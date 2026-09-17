#!/bin/zsh
set -o pipefail
cd "$(dirname "$0")"
LOG="run_log.txt"
: > "$LOG"
echo "=== START $(date) ===" >> "$LOG"

echo "--- [1/3] CA_CNN_learning_77.ipynb (data gen + training) $(date) ---" >> "$LOG"
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=tensorflow-env \
  --ExecutePreprocessor.timeout=-1 \
  CA_CNN_learning_77.ipynb >> "$LOG" 2>&1
RC1=$?
echo "--- learning exit code: $RC1 $(date) ---" >> "$LOG"
if [ $RC1 -ne 0 ]; then
  echo "=== ABORTED after learning failure $(date) ===" >> "$LOG"
  exit $RC1
fi

echo "--- [2/3] PDSI_latin.ipynb $(date) ---" >> "$LOG"
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=tensorflow-env \
  --ExecutePreprocessor.timeout=-1 \
  PDSI_latin.ipynb >> "$LOG" 2>&1
RC2=$?
echo "--- latin exit code: $RC2 $(date) ---" >> "$LOG"
if [ $RC2 -ne 0 ]; then
  echo "=== ABORTED after latin failure $(date) ===" >> "$LOG"
  exit $RC2
fi

echo "--- [3/3] PDSI_midpoint.ipynb $(date) ---" >> "$LOG"
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=tensorflow-env \
  --ExecutePreprocessor.timeout=-1 \
  PDSI_midpoint.ipynb >> "$LOG" 2>&1
RC3=$?
echo "--- midpoint exit code: $RC3 $(date) ---" >> "$LOG"

echo "=== ALL DONE $(date) ===" >> "$LOG"
