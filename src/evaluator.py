import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(pipeline, X_test, y_test):
    """
    훈련된 모델의 성능을 평가하고 결과를 출력/시각화합니다.
    
    Args:
        pipeline: 훈련된 모델 파이프라인
        X_test: 테스트용 피처 데이터
        y_test: 테스트용 타겟 데이터
    """
    print(" [evaluator] 모델 성능 평가 시작...")
    
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n--- 모델 성능 평가 결과 ---")
    print(f"  - 정확도 (Accuracy):  {accuracy:.4f}")
    print(f"  - 정밀도 (Precision): {precision:.4f}")
    print(f"  - 재현율 (Recall):    {recall:.4f}")
    print(f"  - F1-Score:          {f1:.4f}")
    print("---------------------------\n")

    # 혼동 행렬 시각화
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['유지 (No)', '이탈 (Yes)'],
                yticklabels=['유지 (No)', '이탈 (Yes)'])
    plt.title('Confusion Matrix') #혼동 행렬
    plt.xlabel('Predicted Label') # 예측된 레이블
    plt.ylabel('True Label') # 실제 레이블
    
    print("✅ [evaluator] 혼동 행렬 그래프를 확인하세요.")
    plt.show()
