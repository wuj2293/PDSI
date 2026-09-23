# 데이터 요구사항

이 저장소(`PDSI`)에는 코드만 있습니다. 실행에 필요한 데이터는 **저장소 밖, 형제 폴더인
`DATA/`**에 있어야 합니다(용량이 커서 git으로 관리하지 않습니다). 즉 다음과 같은 구조를
기대합니다.

```
어떤-폴더/
├── PDSI/     <- 이 저장소 (git clone)
└── DATA/     <- 아래 목록의 데이터 (별도로 전달받거나 직접 생성)
```

기본적으로 각 노트북은 `PDSI` 폴더를 자동으로 찾아 그 형제 폴더 `DATA`를 씁니다. 다른 위치에
`DATA`를 두고 싶다면 환경변수 `PDSI_DATA_ROOT`를 `DATA`를 담고 있는 상위 폴더 경로로 지정하세요.

```bash
export PDSI_DATA_ROOT="/path/to/그폴더"   # 이 아래에 DATA/ 가 있어야 함
```

## 필요한 파일 목록

| 경로 | 내용 | 어디서 오나 |
|---|---|---|
| `DATA/CA_77/models/model_1.keras` ~ `model_5.keras` | 학습된 CNN 앙상블(5개) | `code/CNN/CA_77/CA_CNN_learning_77.ipynb` 실행 (수 시간 소요) |
| `DATA/SDM_data/latin/latlong_ex.xlsx` | 격자 좌표 ↔ 행정구역(SIG_ENG_NM) 매핑 | SDM 원본 데이터(별도 전달) |
| `DATA/SDM_data/midpoint/<species>/ssp{126,245,585}_{2030,2050,2070,2090}.csv` (12개) | 시나리오·시기별 종분포모델(SDM) 확률 격자 | `code/SDM/SDM.ipynb` / `process_suitability_maps.ipynb`의 산출물 |
| `DATA/SDM_data/latin/local_index.csv` | 행정구역 목록(현재 167개) | `code/SDM/find_local.ipynb` 실행 |
| `DATA/SDM_data/latin/<species>/sampling.pkl` | 행정구역×시나리오×시기별 20×20(400칸) LHS 샘플 확률 격자 | `code/SDM/find_local.ipynb` 실행 |
| `DATA/SDM_data/latin/<species>/all.pkl` | 위와 같지만 샘플링 전 전체 격자(진단용 그림에만 씀, SI 계산엔 불필요) | `code/SDM/find_local.ipynb` 실행 |
| `DATA/weights/WeightByInitial_new.csv` | 규칙(rule)×초기값(initial)별 60세대 시점 보정된 lookup 값 | `code/Weight/compute_weight_by_initial.py` 실행(약 5~6분) |
| `DATA/Results/` | 계산 결과·그림 출력 폴더 | 노트북이 자동 생성(입력 아님) |

`<species>` 자리는 실제로 쓰는 SDM 산출물 이름입니다(예: `TES_maxent`). 지금 이 프로젝트가 쓰는
값은 `DATA/SDM_data/latin/TES_maxent/`이고, `find_local.ipynb`의 `species` 변수를 그 이름으로
맞춰서 실행하면 됩니다.

## 참고

- 로컬 환경에는 위 12개 원본 SDM CSV가 현재 `DATA/`가 아니라 `PDSI_data/SDM_data/midpoint/TES_ensemble/`(약 80MB, 다른 종/방법 이름)에만 있습니다. `find_local.ipynb`를 `TES_ensemble` 종으로 그대로 돌리면 검증되지만, 실제 파이프라인이 쓰는 `TES_maxent` 원본 CSV는 별도로 확보해야 합니다.
- `DATA/CA_77/models/*.keras`, `DATA/weights/WeightByInitial_new.csv`, `DATA/SDM_data/latin/TES_maxent/{sampling,all}.pkl`, `DATA/SDM_data/latin/local_index.csv`는 이미 만들어져 있으면 위 생성 노트북/스크립트를 다시 돌릴 필요가 없습니다.
