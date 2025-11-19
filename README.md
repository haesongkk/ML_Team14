# 통신사 고객 이탈 예측 모델 MVP 개발

## 1. 프로젝트 목표

- **핵심 목표**: 통신사 고객 데이터를 분석하여 고객 이탈을 예측하는 머신러닝 모델의 **MVP(Minimum Viable Product, 최소 기능 제품) 파이프라인**을 구축합니다.
- **개인 목표**: 데이터 분석의 전체 흐름(A to Z)을 직접 경험하고, 데이터 기반의 문제 해결 능력을 기르는 것을 목표로 합니다.

## 2. 데이터셋

- **이름**: IBM Telco Customer Churn
- **출처**: Kaggle
- **링크**: [https://www.kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **획득 방식**: `kagglehub` API를 사용하여 코드로 다운로드

## 3. MVP 개발 워크플로우 (작업 흐름)

이 프로젝트는 아래의 5단계 순서로 진행합니다. 각 단계는 독립된 노트북 파일에서 실험하고, 최종적으로는 전체 프로세스가 연결되도록 구성합니다.

### 0단계: 환경 설정 (Environment Setup)

- **목표**: 프로젝트를 시작할 수 있는 개발 환경을 구축합니다.
- **수행 작업**:
    1.  Git 저장소를 로컬에 `clone` 합니다.
    2.  `requirements.txt` 파일에 필요한 라이브러리(`pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `kagglehub` 등)를 명시하고, 가상환경에 설치합니다. (`pip install -r requirements.txt`)
    3. data/01_raw 아래에 있는 파일은 Kaggle에서 다운 받은 원본 데이터입니다.

### 1단계: 데이터 로딩 및 기본 탐색 (Data Loading & Basic Exploration)

- **목표**: 데이터를 성공적으로 불러와 구조와 기본 정보를 파악합니다.
- **위치**: `notebooks/01_data_exploration.ipynb`
- **수행 작업**:
    1.  `pandas`를 사용해 CSV 파일을 데이터프레임으로 불러옵니다.
    2.  `.info()`, `.head()`, `.describe()`, `.isnull().sum()` 등을 통해 데이터의 전체적인 구조, 타입, 결측치 여부 등을 확인합니다.

### 2단계: 데이터 탐색 및 시각화 (EDA & Visualization)

- **목표**: 데이터의 특징을 깊이 이해하고, 이탈(Churn)에 영향을 미치는 변수에 대한 가설을 세웁니다.
- **위치**: `notebooks/01_data_exploration.ipynb`
- **수행 작업**:
    1.  `seaborn`, `matplotlib`을 이용해 주요 변수들의 분포를 시각화합니다.
    2.  **핵심 분석**: 이탈(Churn) 여부에 따라 다른 변수들이 어떻게 달라지는지 비교 분석합니다. (예: 계약 기간별 이탈률, 인터넷 서비스 종류별 이탈률 등)
    3.  분석 결과를 바탕으로 **2~3개 이상의 유의미한 인사이트(가설)**를 마크다운으로 기록합니다.

### 3단계: 데이터 전처리 및 피처 엔지니어링 (Preprocessing & Feature Engineering)

- **목표**: 모델이 학습할 수 있는 형태로 데이터를 가공하고 정제합니다.
- **위치**: `notebooks/02_feature_engineering.ipynb`
- **수행 작업**:
    1.  불필요한 컬럼을 제거합니다. (예: `customerID`)
    2.  결측치를 처리합니다. (MVP 단계에서는 간단히 채우거나 해당 행을 제거)
    3.  범주형(Categorical) 변수를 숫자로 변환합니다. (예: `Yes`/`No` -> `1`/`0`, One-Hot Encoding 등)
    4.  전체 데이터를 **훈련(Train) 데이터**와 **테스트(Test) 데이터**로 분리합니다. (`train_test_split`)

### 4단계: 모델링 및 평가 (Modeling & Evaluation)

- **목표**: 머신러닝 모델을 학습시키고, 객관적인 성능을 평가합니다.
- **위치**: `notebooks/03_modeling_and_evaluation.ipynb`
- **수행 작업**:
    1.  **기준 모델(Baseline Model)**로 `LogisticRegression`을 학습시킵니다.
    2.  **개선 모델**로 `RandomForestClassifier`를 학습시킵니다.
    3.  Test 데이터셋을 이용해 각 모델의 성능을 평가합니다. (주요 평가지표: **Accuracy**, **F1-Score**)
    4.  두 모델의 성능을 비교하고, 어떤 모델이 이 문제에 더 적합한지 결론을 내립니다.

### 5단계: 결과 해석 (Result Interpretation)

- **목표**: 모델의 예측 결과를 해석하여 '왜' 그런 예측이 나왔는지 이해하고, 비즈니스 인사이트를 도출합니다.
- **위치**: `notebooks/03_modeling_and_evaluation.ipynb`
- **수행 작업**:
    1.  `RandomForest` 모델의 `feature_importances_` (변수 중요도)를 시각화합니다.
    2.  고객 이탈에 가장 큰 영향을 미치는 상위 5개 변수를 확인하고, 그 의미를 해석하여 마크다운으로 정리합니다.
    3.  이를 바탕으로 **간단한 이탈 방지 전략 아이디어**를 1~2가지 제안하며 마무리합니다.

## 4. 프로젝트 실행 방법

1.  `requirements.txt`의 라이브러리를 설치합니다.
    ```bash
    pip install -r requirements.txt
    ```
2.  `notebooks/` 폴더의 주피터 노트북을 `01`부터 순서대로 실행합니다.

---