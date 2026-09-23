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
├── DATA/     <- 2번(생성된 데이터): 모델, weight, 샘플링 결과 등
└── SDM/      <- 1번(원본 데이터): 종 발견지점, 환경변수 래스터
```

`DATA`는 기본적으로 `PDSI`의 형제 폴더로 자동 탐색됩니다(환경변수 `PDSI_DATA_ROOT`로 다른 위치
지정 가능, `code/CNN/CA_77`·`code/SDM/find_local.ipynb`에 적용됨). `SDM/` 폴더는 `SDM.ipynb`,
`process_suitability_maps.ipynb`가 참조하는데, **이 두 노트북은 아직 하드코딩된 다른 사용자
경로(`/Users/mkim/...`)를 그대로 쓰고 있어서 실행 전에 직접 경로를 고쳐야 합니다** (이번에는
이 두 파일의 경로는 손대지 않았습니다).

## 1. 원본 데이터 (코드로 생성 불가)

| 경로 | 내용 | 현재 로컬에 있는 위치 |
|---|---|---|
| `SDM/붉은귀거북/붉은귀거북_생태원&생물학과&GBIF.csv` | 붉은귀거북 발견 지점(위경도) 실측 기록 | 있음 |
| `SDM/env/current/*.asc` | 현재 기후 환경변수 래스터(bio_1, bio_2, slope, river 등) | 있음 (~39MB) |
| `SDM/env/ssp{126,245,585}_{2030,2050,2070,2090}/*.asc` (12세트) | 미래 기후 시나리오별 환경변수 래스터 | 있음 (세트당 ~38MB, 총 ~500MB) |
| `DATA/SDM_data/latin/latlong_ex.xlsx` | 격자 좌표 ↔ 행정구역(SIG_ENG_NM) 매핑표 | 있음 |

이 4가지는 실측 데이터/외부에서 받은 GIS 자료라 저장소의 어떤 코드로도 다시 만들 수 없습니다.
분실하면 안 됩니다.

## 2. 생성된 데이터 (코드가 만들어냄)

| 경로 | 내용 | 만드는 코드 | 입력 |
|---|---|---|---|
| `DATA/SDM_data/midpoint/<species>/ssp*.csv` (12개) | 시나리오·시기별 서식지 적합성 확률(격자) | `SDM.ipynb`(MaxEnt 학습·예측, `.tif` 생성) → `process_suitability_maps.ipynb`(`.tif`→`.csv` 변환) | 위 1번 원본 데이터 |
| `DATA/SDM_data/latin/local_index.csv`, `<species>/{sampling,all}.pkl` | 행정구역별 400칸 LHS 샘플 | `find_local.ipynb` **또는** `process_suitability_maps.ipynb`(둘 다 같은 산출물을 만듦, 아래 참고) | `ssp*.csv` 12개 + `latlong_ex.xlsx` |
| `DATA/CA_77/models/model_1~5.keras` | 학습된 CNN 앙상블 | `code/CNN/CA_77/CA_CNN_learning_77.ipynb`(자체 CA 시뮬레이션으로 학습 데이터도 생성, 수 시간 소요) | 없음(완전 자체 생성) |
| `DATA/weights/WeightByInitial_new.csv` | 규칙(rule)×초기값(initial)별 60세대 시점 보정 lookup 값 | `code/Weight/compute_weight_by_initial.py`(약 5~6분) | 없음(완전 자체 생성) |
| `DATA/Results/` | 계산 결과·그림 | type1/type2 노트북이 자동 생성 | 위 전부 |

**⚠️ 중복 발견:** `find_local.ipynb`와 `process_suitability_maps.ipynb`가 둘 다 각자
"`ssp*.csv` → 행정구역별 LHS 400 샘플링 → `all.pkl`/`sampling.pkl`/`local_index.csv` 저장"을
수행합니다. 어느 쪽이 실제로 쓰는 파이프라인인지, 왜 두 벌이 있는지는 확인이 필요합니다.

## 요약: 지금 당장 있어야 하는 것 vs 없어도 되는 것

- **`latlong_ex.xlsx`만 있으면 된다는 건 아닙니다.** `ssp*.csv`(서식지 적합성)까지 만들려면
  `SDM/`의 발견지점 데이터와 환경변수 래스터 13세트가 먼저 있어야 합니다. 다행히 로컬에 이미
  있습니다.
- 반대로 **CNN 모델**과 **weight 테이블**은 정말로 코드만으로(외부 데이터 없이) 처음부터
  다시 만들 수 있습니다.
- `DATA/`에 이미 만들어진 산출물(모델, weight, `sampling.pkl` 등)이 있다면, 그걸 다시 만드는
  단계는 전부 건너뛰고 `PDSI_newweight_81scenarios_regions.ipynb`부터 바로 실행하면 됩니다.
