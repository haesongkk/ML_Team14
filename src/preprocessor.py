from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def create_preprocessor_and_split_data(df):
    """
    데이터를 훈련/테스트용으로 나누고, 전처리 파이프라인을 생성합니다.
    
    Args:
        df (pandas.DataFrame): 정제된 데이터프레임
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, preprocessor)
    """
    print("[preprocessor] 데이터 전처리 시작...")
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # stratify=y 옵션으로 훈련/테스트 데이터의 Churn 비율을 원본과 유사하게 유지
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [col for col in X.columns if col not in numeric_features]

    # 전처리기(preprocessor) 정의
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    print("✅ [preprocessor] 데이터 분리 및 전처리기 생성 완료")
    return X_train, X_test, y_train, y_test, preprocessor
