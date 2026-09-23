# PDSI

## 개요

CA(Cellular Automata) + CNN 기반으로 외래종 확산 위험도 지표인 **PDSI**(Pathway-Dependent
Spread Intensity)를 계산하는 프로젝트입니다. 77개 CA 규칙(rule)에 대해 학습한 CNN 5개로,
행정구역·기후 시나리오별 확산 분포를 추정하고 이를 하나의 지표(SI/PDSI)로 요약합니다.

## 환경 설정

```bash
git clone https://github.com/wuj2293/PDSI.git
cd PDSI
pip install -r requirements.txt
```

**데이터**는 이 저장소에 들어있지 않습니다. `DATA_REQUIREMENTS.md`를 먼저 읽고, 저장소의
형제 폴더로 `DATA/`를 준비하세요(위치가 다르면 환경변수 `PDSI_DATA_ROOT`로 지정).

Jupyter 커널은 위 `requirements.txt`가 설치된 환경(로컬 가상환경 또는 Colab)이면 됩니다.
CNN 학습(`CA_CNN_learning_77.ipynb`)은 GPU가 있으면 훨씬 빠릅니다.

## 폴더 구조

- `code/CNN/CA_77/`
  - `CA_CNN_learning_77.ipynb` — CA 77규칙 학습 데이터 생성 + CNN 5개 학습(수 시간 소요)
  - `PDSI_newweight_81scenarios_regions.ipynb`,
    `PDSI_newweight_81scenarios_extra15regions.ipynb` — 81개 기후 시나리오 × 100회 반복으로
    지역별 PDSI(SI)를 계산. **type1**(cell-wise, 이진화 난수를 4개 시기마다 독립적으로 뽑음)과
    **type2**(global, 이진화 난수를 위치별로 뽑아 4개 시기가 공유)를 함께 계산·저장합니다.
- `code/SDM/`
  - `SDM.ipynb` — 발견 지점 + 환경변수 래스터로 MaxEnt 학습, 시나리오별 서식지 적합도
    GeoTIFF(`.tif`) 생성
  - `process_suitability_maps.ipynb` — 그 `.tif`를 CSV로 변환하고, 행정구역별로 나눈 뒤
    라틴 하이퍼큐브 샘플링(LHS)으로 20×20(400칸)을 뽑아 `sampling.pkl`/`all.pkl`/
    `local_index.csv`로 저장하는 정식 파이프라인(이전에 있던, 같은 일을 하던
    `find_local.ipynb`는 삭제했습니다)
- `code/Weight/`
  - `eca_core.py`, `compute_weight_by_initial.py`, `build_regression77_docx.py` — CA 규칙별
    "초기값 → 60세대 시점 셀 개수" 보정 lookup 테이블(`WeightByInitial_new.csv`)을 만드는 코드.
    자세한 배경은 `code/Weight/README_WeightByInitial_correction.md` 참고.
- `arc_fig.ipynb` — PDSI 결과 시각화·도표 생성

## 실행 순서

이미 `DATA/`에 필요한 파일이 다 있다면(→ `DATA_REQUIREMENTS.md`) 1~3번은 건너뛰고 바로
`PDSI_newweight_81scenarios_regions.ipynb`부터 실행하면 됩니다.

1. `code/SDM/SDM.ipynb` → `code/SDM/process_suitability_maps.ipynb` — 원본 발견 지점·환경변수
   래스터로부터 `sampling.pkl`/`local_index.csv` 생성
2. `code/Weight/compute_weight_by_initial.py` — `WeightByInitial_new.csv` 생성
   ```bash
   cd code/Weight
   python3 compute_weight_by_initial.py   # 77규칙 전체, 약 5~6분
   ```
3. `code/CNN/CA_77/CA_CNN_learning_77.ipynb` — CNN 모델 5개 학습
4. `code/CNN/CA_77/PDSI_newweight_81scenarios_regions.ipynb` (+
   `PDSI_newweight_81scenarios_extra15regions.ipynb`) — type1/type2 PDSI 계산. 결과는
   `DATA/Results/`에 저장됩니다.

## 참고

- `code/CNN/CA_77/`, `code/SDM/`에는 이 목록보다 많은 실험용 노트북이 로컬에 있을 수 있지만,
  현재 검증되어 저장소에 올라간 것은 위에 적힌 파일들뿐입니다.
- 이전에 쓰던 `weight_generate.ipynb`/`weight_figure.ipynb`(회귀 기반, 편향 있음)는 삭제되었고
  `code/Weight/README_WeightByInitial_correction.md`에 배경이 남아 있습니다.
