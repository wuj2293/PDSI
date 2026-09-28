# 데이터 요구사항

이 저장소(`PDSI`)에는 코드만 있습니다. 실행에 필요한 데이터는 저장소 밖에 있어야 하고,
**두 종류**로 나뉩니다.

1. **진짜 원본(raw) 데이터** — 코드로 만들 수 없고 반드시 그대로 있어야 합니다.
2. **생성된 데이터** — 1번을 입력으로 저장소 안의 코드가 만들어내는 중간/최종 산출물입니다.
   이미 만들어져 있으면 해당 노트북을 다시 돌릴 필요가 없습니다.

## 폴더 구조

```
어떤-폴더/
├── PDSI/     <- 이 저장소 (git clone)
└── DATA/     <- 1번(원본: MaxEnt 지도, 행정구역 매핑표) + 2번(생성된 데이터: 모델, weight, 샘플링 결과 등)
```

`DATA`는 기본적으로 `PDSI`의 형제 폴더로 자동 탐색됩니다(환경변수 `PDSI_DATA_ROOT`로 다른 위치
지정 가능). `code/CNN/CA_77`, `code/SDM/make_all_pkl.ipynb`, `code/SDM/make_sampling_pkl.ipynb` 모두
같은 방식으로 경로를 찾습니다.

## 1. 원본 데이터 (코드로 생성 불가)

| 경로 | 내용 | 현재 로컬에 있는 위치 |
|---|---|---|
| `DATA/SDM_data/Maxent/ssp{126,245,585}_{2030,2050,2070,2090}.csv` (12개) | 붉은귀거북 서식지 적합성 확률(1 km 격자, 열 `x, y, {시나리오}`) — **외부에서 받은 MaxEnt 3.4.1 결과** (`Trachemys_scripta_elegans_{시나리오}_avg.asc`를 격자점 CSV로 변환한 것, 2025-03). CA 입력으로 쓰는 지도 | 있음 |
| `DATA/SDM_data/latin/latlong_ex.xlsx` | 격자 좌표 ↔ 행정구역(SIG_ENG_NM) 매핑표 | 있음 |

이 2가지는 외부에서 받은 자료라 저장소의 어떤 코드로도 다시 만들 수 없습니다. 분실하면 안 됩니다.
MaxEnt 모델 자체(발견 지점, 환경변수, 실행 설정)는 이 저장소 범위 밖입니다.

## 2. 생성된 데이터 (코드가 만들어냄)

| 경로 | 내용 | 만드는 코드 | 입력 |
|---|---|---|---|
| `DATA/SDM_data/latin/local_index.csv`, `TES_maxent/all.pkl` | 행정구역별 전체 셀 | `code/SDM/make_all_pkl.ipynb` | `Maxent/ssp*.csv` 12개 + `latlong_ex.xlsx` |
| `DATA/SDM_data/latin/TES_maxent/sampling.pkl` | 행정구역별 400칸 1차원 LHS 샘플(중복 제거) | `code/SDM/make_sampling_pkl.ipynb` | `all.pkl` |
| `DATA/CA_77/models/model_1~5.keras` | 학습된 CNN 앙상블 | `code/CNN/CA_77/CA_CNN_learning_77.ipynb`(자체 CA 시뮬레이션으로 학습 데이터도 생성, 수 시간 소요) | 없음(완전 자체 생성) |
| `DATA/weights/WeightByInitial_new.csv` | 규칙(rule)×초기값(initial)별 60세대 시점 보정 lookup 값 | `code/Weight/compute_weight_by_initial.py`(약 5~6분) | 없음(완전 자체 생성) |
| `DATA/Results/` | 계산 결과·그림 | type1/type2 노트북이 자동 생성 | 위 전부 |

**샘플링 방식:** `make_sampling_pkl.ipynb`는 래스터 순서(북→남, 서→동)로 정렬한 셀에 **셀 단위 1차원 LHS**를 적용합니다.
후보 ≥ 400이면 셀을 400개 연속 묶음으로 나눠 묶음마다 하나씩 뽑아 중복이 없고(큰 묶음 위치는 무작위로 해 모든 셀의 추출 확률을 같게 함),
후보 < 400이면 모든 셀을 ⌊400/n⌋~⌈400/n⌉번 고르게 씁니다. 20×20 배치는 무작위, 난수는 `SEED=0` + 지역 이름(md5)으로 고정합니다.
위치는 지역마다 한 번만 정하므로 모든 시기·시나리오에서 같은 400개 지점이 같은 칸에 쓰입니다.

- 기존(제출 논문) 샘플 `TES_maxent/archive/sampling_1D_original.pkl`: 같은 1차원 LHS지만 연속값을 정수로 잘라 중복·누락이 있었고 시드 미고정.
- 2Dfixed 샘플 `TES_maxent/archive/sampling_2Dfixed_20260926.pkl`(x·y 순위 공간 2D 층화): 중복은 없지만 불규칙한 지역에서 경계 셀을 과하게 뽑아
  평균 적합도가 치우쳐(시드 20개 검증, 34개 중 19개 지역, 최대 0.027) 쓰지 않음. 그 PDSI 결과는 `DATA/Results/archive_2D_sampling_20260926/`.

**제출 논문과의 관계:** 제출본 Supplementary Data 2의 PDSI는 `CA_old/2025 논문용/CA2025/슈퍼컴_data/76_SI_data_latin.csv`와
값이 모두 같고, 그 계산의 `initial` 값은 위 `Maxent/ssp*.csv` + 기존 1D `sampling.pkl`
(`PDSI_data/SDM_data/latin/TES_maxent/`에 보존)로 계산한 기대값과 맞습니다. 즉 제출 결과도 같은 MaxEnt 지도를 썼습니다.

## 요약: 지금 당장 있어야 하는 것 vs 없어도 되는 것

- **꼭 있어야 하는 원본은 두 가지뿐입니다:** MaxEnt 지도(`Maxent/ssp*.csv` 12개)와 `latlong_ex.xlsx`.
  이 둘로 `all.pkl`과 `sampling.pkl`을 만들 수 있습니다.
- 반대로 **CNN 모델**과 **weight 테이블**은 정말로 코드만으로(외부 데이터 없이) 처음부터
  다시 만들 수 있습니다.
- `DATA/`에 이미 만들어진 산출물(모델, weight, `sampling.pkl` 등)이 있다면, 그걸 다시 만드는
  단계는 전부 건너뛰고 `PDSI_newweight_81scenarios_regions.ipynb`부터 바로 실행하면 됩니다.
