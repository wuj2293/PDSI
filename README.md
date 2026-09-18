# PDSI

## 개요

이 프로젝트는 PDSI 관련 분석과 모델링 실험을 담고 있습니다. 주요 워크플로는 다음과 같습니다.

- `code/CNN`: Cellular Automata 기반 PDSI 데이터 생성, CNN 분류/학습/평가
- `code/SDM`: Species distribution modeling (SDM) 및 적합도 지도 생성
- `code/Weight`: 가중치 생성과 시각화
- `arc_fig.ipynb`: PDSI 관련 결과 시각화 및 도표 생성

> 참고: `code/CNN/README.md`에는 고수준 파이프라인 정보가 있지만, 현재 코드가 많이 수정되어 있어 실제 실행 경로는 이 루트 README와 서브폴더 내 노트북을 함께 확인하는 것이 좋습니다.

---

## 폴더 구조

- `code/CNN`은 `CA_*` 하위 폴더별로 서로 다른 조건(예: CA 크기, PDSI 추출 방식)을 다룹니다.
- `code/SDM`은 SDM 분석과 적합도 지도 생성 전용입니다.
- `code/Weight`는 가중치 생성/분석 및 시각화 전용입니다.

---

## 실행 순서

1. `PDSI_data` 폴더 위치 확인

   대부분 노트북이 프로젝트 루트의 상위에 `PDSI_data` 폴더가 있다고 가정합니다.

   예:
   ```text
   /Users/mkim/Library/CloudStorage/GoogleDrive-wuj2293@gmail.com/My Drive/Research/PDSI
   /Users/mkim/Library/CloudStorage/GoogleDrive-wuj2293@gmail.com/My Drive/Research/PDSI_data
   ```

   노트북 상단에서 `PROJECT_ROOT`와 `DATA_ROOT` 경로를 먼저 확인하고 필요하면 수정하세요.

2. 주요 실행 대상

   - `code/CNN/CA_152/CA_CNN_learning_152.ipynb` 또는 `code/CNN/CA_76/...` 등의 CNN 학습 노트북
   - `code/CNN/CA_152/PDSI_latin.ipynb`, `PDSI_midpoint.ipynb` 등 PDSI 계산/분석 노트북
   - `code/SDM/SDM.ipynb` 또는 `process_suitability_maps.ipynb`
   - `code/Weight/weight_generate.ipynb`, `weight_figure.ipynb`

3. 노트북 실행 방식

   - 로컬 환경: `jupyter notebook` 또는 `jupyter lab`
   - Google Colab: 필요 시 경로와 드라이브 마운트 설정을 변경
   - GPU가 있으면 CNN 학습 속도 향상에 유리

4. 경로 설정 확인

   핵심은 `base / "PDSI_data"` 또는 `DATA_ROOT` 경로가 실제 데이터 위치를 가리키는지 확인하는 것입니다.
   대부분 노트북은 다음과 같은 경로를 사용합니다.
   - `PDSI_data/CA_152`
   - `PDSI_data/SDM_data`
   - `PDSI_data/weights`
   - `PDSI_data/Results`

---

## 데이터 및 의존성

### 데이터

- `PDSI_data` 폴더는 코드와 별개로 외부에 보관됩니다.
- `CNN` 노트북은 학습 데이터, 모델 파일, 체크포인트를 `PDSI_data/CA_*` 아래에 생성/사용합니다.
- `SDM`과 `Weight` 노트북은 `PDSI_data/SDM_data`, `PDSI_data/weights` 등을 참조합니다.

### 추천 환경

- Python 3.11 또는 3.10
- Jupyter Notebook / Jupyter Lab
- 주요 패키지:
  - `numpy`, `pandas`, `matplotlib`, `seaborn`
  - `tensorflow` / `keras`
  - `h5py`
  - `scikit-learn`
  - `geopandas` / `rasterio` (SDM 관련 시 사용 가능)

---

## 코드 흐름 요약

### CNN

- 데이터 생성 → 모델 학습 → 평가/비교
- 각 `CA_*` 폴더는 서로 다른 CA 크기 또는 PDSI 처리 방식을 실험합니다.
- `CA_CNN_learning_152.ipynb` 등은 전체 파이프라인을 순차적으로 실행합니다.

### PDSI 계산

- `PDSI_latin.ipynb`는 LHS 샘플링 기반 PDSI 계산
- `PDSI_midpoint.ipynb`는 지역 중점 기반 PDSI 계산
- 결과는 `PDSI_data/Results/`에 HDF5/CSV 형식으로 저장될 가능성이 큽니다.

### SDM

- `SDM.ipynb`는 SDM 모델 학습 및 예측 결과 생성
- `process_suitability_maps.ipynb`는 적합도 지도 처리 및 시각화

### Weight

- `weight_generate.ipynb`에서 가중치를 계산
- `weight_figure.ipynb`에서 가중치 결과를 시각화

---

## 참고

- 이 루트 README는 현재 `PDSI` 폴더 구조와 주요 작업 흐름을 정리한 것입니다.
- 실제 실행 전 각 노트북 상단의 경로/환경 설정을 먼저 확인해야 합니다.
- 추가로 필요한 설명은 `code/CNN/README.md`와 각 노트북 설명을 병행해서 확인하세요.
