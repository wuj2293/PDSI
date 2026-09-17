#!/bin/zsh
cd "$(dirname "$0")"
LOG="run_log.txt"
echo "=== RESUME (latin+midpoint) $(date) ===" >> "$LOG"

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
