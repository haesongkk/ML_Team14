# 각 파일에서 필요한 함수들을 가져옵니다.
from data_loader import load_and_clean_data
from preprocessor import create_preprocessor_and_split_data
from model_trainer import train_model
from evaluator import evaluate_model

from preprocessor import create_preprocessor_and_split_data

# 이 코드를 실행하기 전에, 모든 .py 파일이 같은 폴더에 있는지,
# 'WA_Fn-UseC_-Telco-Customer-Churn.csv' 파일도 같은 폴더에 있는지 확인해주세요.

if __name__ == "__main__":
    
    # 1. 데이터 로드 및 정제
    file_path = 'data/01_raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'

    cleaned_df = load_and_clean_data(file_path)
    
    # 데이터 로드에 성공했을 경우에만 다음 단계를 진행합니다.
    if cleaned_df is not None:
        
        # 2. 데이터 전처리 및 분리
        X_train, X_test, y_train, y_test, preprocessor = create_preprocessor_and_split_data(cleaned_df)
        
        # 3. 모델 훈련
        trained_model_pipeline = train_model(X_train, y_train, preprocessor)
        
        # 4. 모델 평가
        evaluate_model(trained_model_pipeline, X_test, y_test)
        
        print("\n🎉 모든 과정이 성공적으로 완료되었습니다!")
