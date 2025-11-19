# [변경점 1] imbalanced-learn의 Pipeline과 SMOTE를 import 합니다.
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier  # RandomForest 대신 XGBoost를 가져옵니다

def train_model(X_train, y_train, preprocessor):
    """
    전처리, SMOTE 오버샘플링, XGBoost 모델을 파이프라인으로 결합하여 훈련시킵니다.
    
    Args:
        X_train: 훈련용 피처 데이터
        y_train: 훈련용 타겟 데이터
        preprocessor: 데이터 전처리기
        
    Returns:
        imblearn.pipeline.Pipeline: 훈련된 모델 파이프라인
    """
    print("⏳ [model_trainer] SMOTE 포함 모델 훈련 시작...")

    # [변경점 2] RandomForest 모델 대신 XGBoost 모델을 정의합니다.
    # XGBoost는 불균형 데이터 처리에 매우 효과적인 파라미터들을 제공합니다.

    # 이탈 고객(1)의 비율이 낮으므로, 'scale_pos_weight'로 가중치를 줍니다.
    # (유지 고객 수 / 이탈 고객 수) 비율을 계산하여 적용하면 좋습니다. (대략 73/27 ≈ 2.7)

    xgb_classifier = XGBClassifier(
        n_estimators=100,      # 부스팅 단계 수 (트리의 개수)
        learning_rate=0.1,     # 학습률
        max_depth=3,           # 트리의 최대 깊이 (과적합 방지)
        random_state=42,
        n_jobs=-1
    )

    # [변경점 3] 전처리기, SMOTE, 모델을 파이프라인으로 연결합니다.
    # 파이프라인은 각 단계를 순서대로 실행합니다.
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb_classifier)
    ])

    # 모델 훈련 (파이프라인이 알아서 SMOTE를 훈련 데이터에만 적용합니다)
    model_pipeline.fit(X_train, y_train)
    
    print("✅ [model_trainer] 모델 훈련 완료")
    return model_pipeline