"""샘플링 시드 민감도 테스트: 한 지역·한 시나리오에서 400셀 샘플링(셀 선택 + 20×20 배치)을
시드마다 다시 뽑아 PDSI(type1/type2, 100회 반복)가 얼마나 흔들리는지 잰다.

  모드 full   : 시드마다 셀 선택과 배치를 모두 새로 (make_sampling_pkl.ipynb 와 같은 방법, SEED 만 바꿈)
  모드 layout : 셀 선택은 SEED=0 으로 고정하고 20×20 배치만 시드마다 새로
  참고        : 기존 1D(제출 논문) / 2Dfixed / 현재 sampling.pkl 샘플도 같은 조건으로 계산

PDSI 계산 함수(모델·weight·clu_SI·이진화)는 PDSI_newweight_81scenarios_regions.ipynb 의 셀을 그대로 실행해 쓰고,
샘플링 함수는 code/SDM/make_sampling_pkl.ipynb 의 셀을 그대로 실행해 쓴다.

  python sensitivity_sampling_seed.py --region Dongducheon-si --scenario 245_245_245_245 --n-seeds 20
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('--region', default='Dongducheon-si')
p.add_argument('--scenario', default='245_245_245_245')
p.add_argument('--n-seeds', type=int, default=20)
p.add_argument('--n-rep', type=int, default=100)
args = p.parse_args()


def cells_of(nb_path, upto_marker):
    cells = json.load(open(nb_path))['cells']
    out = []
    for c in cells:
        src = ''.join(c['source'])
        if upto_marker in src:
            break
        if c['cell_type'] == 'code':
            out.append(src)
    return out


# PDSI 계산부: 노트북 셀 0~9 (지역 목록/출력 파일 셀과 계산 루프는 제외)
ns = {'__file__': str(HERE / 'PDSI_newweight_81scenarios_regions.ipynb')}
import os
os.chdir(HERE)
for src in cells_of(HERE / 'PDSI_newweight_81scenarios_regions.ipynb', 'done_c = set('):
    if src.lstrip().startswith('regions = ['):
        continue
    exec(src, ns)
base = ns['base']

# 샘플링 함수: make_sampling_pkl.ipynb 의 함수 셀
sdm_nb = HERE.parent.parent / 'SDM' / 'make_sampling_pkl.ipynb'
fn_src = next(''.join(c['source']) for c in json.load(open(sdm_nb))['cells']
              if 'def lhs_1d_positions' in ''.join(c['source']))
import hashlib
ns["hashlib"] = hashlib
exec(fn_src, ns)
district_rng, lhs_1d_positions, NUM = ns['district_rng'], ns['lhs_1d_positions'], ns['NUM_SAMPLES']

region, sc = args.region, args.scenario.split('_')
periods = [f'ssp{a}_{y}' for a, y in zip(sc, [2030, 2050, 2070, 2090])]
local_all = ns['local_all']
cells = {s: local_all[s][region] for s in periods}
n = len(cells[periods[0]])


def maxent_mat_from(values_by_period):
    m = np.stack([np.reshape(v, (20, 20)) for v in values_by_period], axis=-1)
    return m.reshape(1, 20, 20, 4)


def mat_from_positions(pos):
    return maxent_mat_from([cells[s][s].values[pos] for s in periods])


def mat_from_sampling(pkl):
    smp = pickle.load(open(pkl, 'rb'))
    return maxent_mat_from([smp[s][region].iloc[:, 2].values for s in periods])


def pdsi(mat):
    t1 = [ns['clu_SI'](*ns['make_CA_distribution_cellwise'](mat, 77)) for _ in range(args.n_rep)]
    t2 = [ns['clu_SI'](*ns['make_CA_distribution_global'](mat, 77, rep)) for rep in range(args.n_rep)]
    return np.array(t1), np.array(t2)


def summarize(label, seed, mat, t1, t2):
    return dict(sample=label, seed=seed, mean_suitability_2030=float(mat[0, :, :, 0].mean()),
                type1_mean=t1.mean(), type1_sd=t1.std(ddof=1), type2_mean=t2.mean(), type2_sd=t2.std(ddof=1))


out_dir = base / 'DATA' / 'Results' / 'sensitivity_sampling_seed'
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / f'{region}_{args.scenario}.csv'
rows = []
t0 = time.time()

np.random.seed(12345)   # type1(cell-wise) 이진화 난수도 고정해 재현 가능하게
arch = base / 'DATA' / 'SDM_data' / 'latin' / 'TES_maxent'
for label, pkl in [('기존1D(제출논문)', arch / 'archive' / 'sampling_1D_original.pkl'),
                   ('2Dfixed', arch / 'archive' / 'sampling_2Dfixed_20260926.pkl'),
                   ('현재 sampling.pkl(새1D, SEED=0)', arch / 'sampling.pkl')]:
    mat = mat_from_sampling(pkl)
    rows.append(summarize(label, None, mat, *pdsi(mat)))
    print(f'{label}: {time.time() - t0:.0f}s', flush=True)

sel0 = lhs_1d_positions(n, NUM, district_rng(region, 0))            # 셀 선택 고정(SEED=0)
for seed in range(args.n_seeds):
    rng = district_rng(region, seed)
    pos = lhs_1d_positions(n, NUM, rng)
    pos = pos[rng.permutation(NUM)]
    mat = mat_from_positions(pos)
    rows.append(summarize('full', seed, mat, *pdsi(mat)))
    lay = sel0[np.random.default_rng([1000 + seed]).permutation(NUM)]
    mat = mat_from_positions(lay)
    rows.append(summarize('layout', seed, mat, *pdsi(mat)))
    print(f'seed {seed}: {time.time() - t0:.0f}s', flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)

df = pd.DataFrame(rows)
df.to_csv(out, index=False)
print(f'\n지역 {region} (후보 {n}셀), 시나리오 {args.scenario}, 시드 {args.n_seeds}개, 반복 {args.n_rep}회')
print(df[df.seed.isna()][['sample', 'mean_suitability_2030', 'type1_mean', 'type1_sd', 'type2_mean', 'type2_sd']].round(4).to_string(index=False))
for mode in ['full', 'layout']:
    d = df[df['sample'] == mode]
    print(f'[{mode}] 시드 간 SD: type1 {d.type1_mean.std():.4f}, type2 {d.type2_mean.std():.4f} '
          f'(범위 type2 {d.type2_mean.min():.4f}~{d.type2_mean.max():.4f}) | 반복 SD 평균 type2 {d.type2_sd.mean():.4f}')
print(f'저장: {out}  ({time.time() - t0:.0f}s)')
