# CA CNN Classification - Google Colab Notebooks

이 폴더에는 Google Colab에서 실행 가능한 Jupyter notebook들이 포함되어 있습니다.

## 📚 노트북 목록

### 1. [1_generate_data.ipynb](1_generate_data.ipynb)
**데이터 생성 노트북**
- Cellular Automata 시뮬레이션 실행
- 228,000개 샘플 생성 (76 rules × 500 samples × 6 density levels)
- 출력: `data500_50.pickle`
- 예상 시간: ~15-20분

### 2. [2_train_models.ipynb](2_train_models.ipynb)
**모델 학습 노트북**
- 5개의 독립적인 CNN 모델 학습
- 각 모델은 서로 다른 random seed 사용
- 자동 Early Stopping 및 Learning Rate Scheduling
- 학습 그래프 자동 생성
- 예상 시간: ~2-3시간 (GPU 사용 시)

### 3. [CA_CNN_Complete.ipynb](CA_CNN_Complete.ipynb)
**전체 통합 노트북**
- 데이터 생성 + 모델 학습을 한 번에 실행
- 전체 파이프라인 확인용
- 예상 시간: ~3-4시간

## 🚀 사용 방법

### Google Colab에서 실행

1. **Google Drive 마운트**
   - 노트북 첫 셀에서 자동으로 Drive를 마운트합니다
   - 권한 승인 필요

2. **GPU 설정** (권장)
   - 런타임 → 런타임 유형 변경 → GPU 선택

3. **순차적 실행**
   ```
   1단계: 1_generate_data.ipynb 실행 → 데이터 생성
   2단계: 2_train_models.ipynb 실행 → 모델 학습
   ```

   또는
   ```
   CA_CNN_Complete.ipynb 실행 → 전체 파이프라인
   ```

### 로컬 환경에서 실행

1. **Jupyter 설치**
   ```bash
   pip install jupyter notebook
   ```

2. **경로 수정**
   - 각 노트북의 경로 설정 부분 수정
   - Google Drive 마운트 셀 제거

3. **실행**
   ```bash
   jupyter notebook
   ```

## 📁 출력 파일 구조

```
Research/CA/
├── Data/
│   ├── data500_50.pickle          # 생성된 학습 데이터
│   ├── models/
│   │   ├── model_1.keras          # 학습된 모델 1
│   │   ├── model_2.keras          # 학습된 모델 2
│   │   ├── model_3.keras          # 학습된 모델 3
│   │   ├── model_4.keras          # 학습된 모델 4
│   │   ├── model_5.keras          # 학습된 모델 5
│   │   ├── training_history_model_1.png
│   │   ├── training_history_model_2.png
│   │   ├── ...
│   │   └── model_comparison.png   # 모델 비교 그래프
│   └── checkpoints/
│       ├── model_checkpoint_1.weights.h5
│       ├── model_checkpoint_2.weights.h5
│       └── ...
```

## ⚙️ 설정 변경

각 노트북의 Configuration 셀에서 다음을 조정할 수 있습니다:

### 데이터 생성
```python
NUM_SAMPLES_PER_RULE = 500  # 샘플 수
DENSITY_LEVELS = [1, 2, 3, 4, 5, 6]  # 밀도 레벨
```

### 모델 학습
```python
BATCH_SIZE = 100  # 배치 크기
EPOCHS = 100  # 최대 에폭
NUM_MODELS = 5  # 학습할 모델 개수
MODEL_SEEDS = [42, 123, 456, 789, 1011]  # Random seeds
```

## 🎯 학습 결과

일반적으로 다음과 같은 성능을 기대할 수 있습니다:
- **Validation Accuracy**: ~83%
- **Test Accuracy**: ~75-78%
- **Training Time per Model**: ~20-40분 (GPU)

## 💡 팁

1. **GPU 사용 권장**
   - CPU로만 학습 시 10배 이상 시간 소요
   - Colab의 무료 GPU를 활용하세요

2. **세션 타임아웃 주의**
   - Colab 무료 버전은 최대 12시간 세션
   - 중간에 끊기지 않도록 주의

3. **데이터 재사용**
   - 한 번 생성한 데이터는 재사용 가능
   - `GENERATE_NEW_DATA = False`로 설정

4. **체크포인트 활용**
   - 학습 중 best model이 자동 저장됨
   - 중단되어도 checkpoint에서 재개 가능

## 🐛 문제 해결

### 메모리 부족
```python
BATCH_SIZE = 50  # 배치 크기 줄이기
```

### 학습 시간 단축
```python
NUM_SAMPLES_PER_RULE = 100  # 샘플 수 줄이기
EPOCHS = 50  # 에폭 수 줄이기
```

### 경로 오류
- Google Drive 경로를 본인의 폴더 구조에 맞게 수정

## 📞 문의

문제가 발생하면 노트북 내의 각 셀을 순차적으로 실행하며 에러 메시지를 확인하세요.

## 🔗 관련 파일

원본 Python 스크립트는 `CA/code/` 폴더에 있습니다:
- `config.py` - 설정 파일
- `generate_data.py` - 데이터 생성 스크립트
- `train_models.py` - 모델 학습 스크립트
