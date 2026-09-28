#!/bin/zsh
# 34개 지역 PDSI(type1/type2, 81개 시나리오 × 100회)를 여러 프로세스로 나눠 동시에 계산한다.
#
#   ./run_pdsi_parallel.sh <archive_folder_name> [19개 지역 분할 수=2] [15개 지역 분할 수=2]
#
# 1) DATA/Results 의 기존 PDSI 결과(두 노트북 출력)를 DATA/Results/<archive_folder_name>/ 으로 옮긴다
#    (노트북은 결과 CSV에 이미 있는 시나리오를 건너뛰므로, 옛 결과가 남아 있으면 다시 계산되지 않는다).
# 2) 두 노트북의 계산 부분(그림·병합 셀 제외)만 임시 사본으로 만들어, 지역 목록을 나눠(regions[i-1::k]) 동시에 실행한다.
#    각 사본은 결과 파일 이름에 _part{i}of{k} 를 붙여 따로 저장한다. git 에 등록된 노트북은 고치지 않는다.
# 3) 조각 결과를 원래 파일 이름으로 합친 뒤(지역 순서는 원래 목록 순서), 원래 두 노트북을 차례로 실행한다.
#    이때 계산은 모두 건너뛰고 그림 그리기와 19+15 -> 34regions 병합만 한다.
set -o pipefail
cd "$(dirname "$0")"
TAG=${1:?"usage: $0 <archive_folder_name> [parts_19] [parts_15]"}
P19=${2:-2}
P15=${3:-2}
export PDSI_DATA_ROOT=${PDSI_DATA_ROOT:-$(cd ../../../.. && pwd)}
RES="$PDSI_DATA_ROOT/DATA/Results"
LOG="run_pdsi_parallel_log.txt"
WORK=$(mktemp -d)
: > "$LOG"

for i in 1 2 3 4 5; do
  [ -f "$PDSI_DATA_ROOT/DATA/CA_77/models/model_$i.keras" ] || { echo "=== ABORTED: model_$i.keras missing ===" >> "$LOG"; exit 1; }
done

echo "--- archive old results -> $RES/$TAG $(date) ---" >> "$LOG"
mkdir -p "$RES/$TAG"
for f in 77_lhs_repeat100_type1_newweight.csv 77_lhs_repeat100_type2_newweight.csv \
         77_lhs_repeat100_type1_newweight_extra15.csv 77_lhs_repeat100_type2_newweight_extra15.csv \
         77_lhs_repeat100_type1_newweight_34regions.csv 77_lhs_repeat100_type2_newweight_34regions.csv \
         boxplots_by_region_paired_newweight lineplots_by_region_newweight boxplots_by_scenario_newweight; do
  [ -e "$RES/$f" ] && mv "$RES/$f" "$RES/$TAG/" && echo "moved $f" >> "$LOG"
done

# 계산 부분만 담은 조각 노트북 만들기
python3 - "$WORK" "$P19" "$P15" <<'EOF' >> "$LOG" 2>&1 || exit 1
import sys, nbformat as nbf
work, parts = sys.argv[1], {'PDSI_newweight_81scenarios_regions.ipynb': int(sys.argv[2]),
                            'PDSI_newweight_81scenarios_extra15regions.ipynb': int(sys.argv[3])}
for nb_name, k in parts.items():
    nb = nbf.read(nb_name, as_version=4)
    last = next(i for i, c in enumerate(nb.cells) if 'done_c = set(' in ''.join(c.source))
    for i in range(1, k + 1):
        sub = nbf.v4.new_notebook(metadata=nb.metadata)
        sub.cells = [nbf.v4.new_code_cell(''.join(c.source)) for c in nb.cells[:last + 1]]
        cfg = next(c for c in sub.cells if 'out_cellwise = ' in c.source and 'regions = ' in c.source)
        cfg.source += (f"\n\n# [run_pdsi_parallel.sh] 조각 {i}/{k}\n"
                       f"regions = regions[{i - 1}::{k}]\n"
                       f"out_cellwise = out_cellwise.with_name(out_cellwise.stem + '_part{i}of{k}.csv')\n"
                       f"out_global = out_global.with_name(out_global.stem + '_part{i}of{k}.csv')\n"
                       f"print('조각 {i}/{k}:', regions)")
        nbf.write(sub, f"{work}/{nb_name[:-6]}_part{i}of{k}.ipynb")
        print('made', f"{nb_name[:-6]}_part{i}of{k}.ipynb")
EOF

echo "=== PARALLEL START ($P19 + $P15 processes) $(date) ===" >> "$LOG"
caffeinate -i -w $$ &
PIDS=()
for nb in "$WORK"/*_part*.ipynb; do
  ( jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.kernel_name=tensorflow-env \
      --ExecutePreprocessor.timeout=-1 "$nb" > "${nb%.ipynb}.log" 2>&1
    echo "--- $(basename $nb) exit code: $? $(date) ---" >> "$LOG" ) &
  PIDS+=($!)
done
for p in $PIDS; do wait $p; done
if grep -q "exit code: [1-9]" "$LOG"; then
  echo "=== ABORTED: a part failed (logs in $WORK) $(date) ===" >> "$LOG"; exit 1
fi

echo "--- merge parts $(date) ---" >> "$LOG"
python3 - "$RES" "$P19" "$P15" <<'EOF' >> "$LOG" 2>&1 || exit 1
import sys, re, ast, pandas as pd, nbformat as nbf
from pathlib import Path
res = Path(sys.argv[1])
jobs = {'PDSI_newweight_81scenarios_regions.ipynb': (int(sys.argv[2]), '77_lhs_repeat100_{t}_newweight'),
        'PDSI_newweight_81scenarios_extra15regions.ipynb': (int(sys.argv[3]), '77_lhs_repeat100_{t}_newweight_extra15')}
for nb_name, (k, stem) in jobs.items():
    src = ''.join(next(c.source for c in nbf.read(nb_name, as_version=4).cells if 'regions = [' in ''.join(c.source)))
    regions = ast.literal_eval(re.search(r'regions = (\[.*?\])', src, re.S).group(1))
    for t in ['type1', 'type2']:
        parts = [pd.read_csv(res / f"{stem.format(t=t)}_part{i}of{k}.csv") for i in range(1, k + 1)]
        merged = parts[0]
        for p in parts[1:]:
            merged = merged.merge(p, on=['scenario', 'rep'], how='inner', validate='one_to_one')
        assert len(merged) == 8100 and set(merged.columns[2:]) == set(regions), (stem, t, merged.shape)
        merged = merged[['scenario', 'rep'] + regions]
        merged.to_csv(res / f"{stem.format(t=t)}.csv", index=False)
        for i in range(1, k + 1):
            (res / f"{stem.format(t=t)}_part{i}of{k}.csv").unlink()
        print('merged', f"{stem.format(t=t)}.csv", merged.shape)
EOF

for nb in PDSI_newweight_81scenarios_regions.ipynb PDSI_newweight_81scenarios_extra15regions.ipynb; do
  echo "--- $nb (plots / 34regions merge only) $(date) ---" >> "$LOG"
  jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=tensorflow-env \
    --ExecutePreprocessor.timeout=-1 "$nb" >> "$LOG" 2>&1 || { echo "=== ABORTED at $nb $(date) ===" >> "$LOG"; exit 1; }
done
rm -rf "$WORK"
echo "=== ALL DONE $(date) ===" >> "$LOG"
