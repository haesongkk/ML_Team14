# ============================================================
# 0. 기본 설정 & 라이브러리
# ============================================================
import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC


# ============================================================
# 1. 데이터 로드 함수 (telco.csv)
# ============================================================

def load_telco_data(csv_path: Union[Path, str] = Path("datasets/telco.csv")) -> pd.DataFrame:
    """datasets/telco.csv를 로드하고, 타입 정리."""
    df = pd.read_csv(csv_path)

    # 공백/문자 포함 가능: 숫자로 강제 변환
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 타깃을 나중에 0/1로 맵핑할 것이므로 원본은 유지
    return df


# ============================================================
# 2. train / test 분리 (Churn 기반 계층 샘플링)
# ============================================================

def stratified_train_test_split_telco(df: pd.DataFrame, test_size=0.2, random_state=42):
    """Churn 라벨을 기준으로 계층 샘플링하여 train/test 분리."""
    df = df.copy()
    split = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    for train_idx, test_idx in split.split(df, df["Churn"]):
        train_set = df.loc[train_idx]
        test_set = df.loc[test_idx]
    return train_set, test_set


# ============================================================
# 3. 전처리 파이프라인 (수치 + 범주형)
#    - 수치: 결측치 median 대체, 표준화
#    - 범주형: 최빈값 대체 + One-Hot 인코딩
# ============================================================

def build_preprocess_and_model_pipeline_telco(base_estimator=None):
    """수치/범주형 전처리 + 입력된 분류기(estimator)를 포함한 파이프라인 생성."""

    # 파이프라인에서 사용할 특성 목록 정의 (Churn/ID 제외)
    num_attribs = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "SeniorCitizen",
    ]
    cat_attribs = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]

    # 수치형 파이프라인
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # 범주형 파이프라인
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    # 전체 컬럼 변환기
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])

    # 기본 분류기 설정
    model = base_estimator
    if model is None:
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight=None,
        )

    full_model_pipeline = Pipeline([
        ("preprocess", full_pipeline),
        ("model", model),
    ])

    return full_model_pipeline


# ============================================================
# 4. 전체 흐름: 데이터 로드 → 분리 → 학습 → 교차검증 → 테스트 평가
# ============================================================

def main():
    # 1) 데이터 로드
    telco = load_telco_data()
    # print("[INFO] telco head():")
    # print(telco.head())

    # 2) 불필요 컬럼 제거
    if "customerID" in telco.columns:
        telco = telco.drop(columns=["customerID"])

    # 3) train / test 분리 (Churn 기준 계층 샘플링)
    train_set, test_set = stratified_train_test_split_telco(telco)
    print(f"[INFO] train size = {len(train_set)}, test size = {len(test_set)}")

    # 4) 학습용 특성과 라벨 분리
    y_train = train_set["Churn"].map({"No": 0, "Yes": 1}).astype(int)
    X_train = train_set.drop(columns=["Churn"])

    y_test = test_set["Churn"].map({"No": 0, "Yes": 1}).astype(int)
    X_test = test_set.drop(columns=["Churn"])

    # 5) 모델 레지스트리 정의
    models = {
        "logreg": LogisticRegression(max_iter=2000, solver="liblinear"),
        "rf": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "gb": GradientBoostingClassifier(random_state=42),
        "svc_rbf": SVC(kernel="rbf", probability=True, random_state=42),
    }

    # 하이퍼파라미터 탐색 공간(파이프라인 내 모델 스텝 접두사: model__)
    param_spaces = {
        "logreg": {
            "model__C": [0.01, 0.1, 1, 3, 10, 30, 100],
            "model__penalty": ["l1", "l2"],
            "model__class_weight": [None, "balanced"],
        },
        "rf": {
            "model__n_estimators": [200, 300, 400, 500],
            "model__max_depth": [None, 5, 10, 20],
            "model__max_features": ["sqrt", "log2", None],
            "model__min_samples_split": [2, 5, 10],
            "model__class_weight": [None, "balanced"],
        },
        "gb": {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.03, 0.1, 0.2],
            "model__max_depth": [2, 3, 4],
            "model__subsample": [0.8, 1.0],
        },
        "svc_rbf": {
            "model__C": [0.1, 1, 3, 10, 30],
            "model__gamma": ["scale", "auto", 0.01, 0.1],
            "model__class_weight": [None, "balanced"],
        },
    }

    best_name = None
    best_pipeline = None
    best_cv_auc = -np.inf
    results = []

    # 6) 각 모델별 하이퍼파라미터 튜닝(RandomizedSearchCV, roc_auc)
    for name, estimator in models.items():
        print(f"[INFO] Evaluating model: {name}")
        pipeline = build_preprocess_and_model_pipeline_telco(estimator)
        param_dist = param_spaces.get(name, {})
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=min(25, max(1, sum(len(v) for v in param_dist.values()))),
            scoring="roc_auc",
            cv=5,
            n_jobs=-1,
            random_state=42,
            verbose=1,
        )
        search.fit(X_train, y_train)
        mean_auc = float(search.best_score_)
        std_auc = float(np.std(search.cv_results_["mean_test_score"]))
        print(f"  Best params: {search.best_params_}")
        print(f"  Best CV ROC-AUC: mean={mean_auc:.4f}, approx_std={std_auc:.4f}")
        results.append((name, mean_auc, std_auc))
        if mean_auc > best_cv_auc:
            best_cv_auc = mean_auc
            best_name = name
            best_pipeline = search.best_estimator_

    # 7) 베스트 모델로 전체 train 재학습
    assert best_pipeline is not None
    print(f"[INFO] Best model by CV ROC-AUC: {best_name} ({best_cv_auc:.4f})")
    print("[INFO] Fitting best model on full training set ...")
    best_pipeline.fit(X_train, y_train)

    # 8) 테스트 세트 평가
    print("[INFO] Evaluating best model on test set ...")
    test_pred = best_pipeline.predict(X_test)
    if hasattr(best_pipeline.named_steps["model"], "predict_proba"):
        test_proba = best_pipeline.predict_proba(X_test)[:, 1]
        test_roc_auc = roc_auc_score(y_test, test_proba)
    else:
        test_proba = None
        test_roc_auc = np.nan
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred)
    print(f"[RESULT] {best_name} - Test Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}, ROC-AUC: {test_roc_auc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, test_pred, digits=4))

    # 9) 모델 저장 (전처리 + 모델)
    MODEL_PATH = Path("artifacts")
    MODEL_PATH.mkdir(exist_ok=True)
    # 모델별 파일 저장
    model_file = MODEL_PATH / f"telco_{best_name}_pipeline.joblib"
    joblib.dump(best_pipeline, model_file)
    print(f"[INFO] Saved best model pipeline to: {model_file.resolve()}")
    # 베스트 별칭 저장
    alias_file = MODEL_PATH / "telco_best_pipeline.joblib"
    joblib.dump(best_pipeline, alias_file)
    print(f"[INFO] Also saved alias to: {alias_file.resolve()}")


if __name__ == "__main__":
    main()
