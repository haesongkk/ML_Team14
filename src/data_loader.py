import pandas as pd

def load_and_clean_data(file_path):
    """
    CSV 파일에서 데이터를 로드하고 기본적인 정제를 수행합니다.
    - TotalCharges 결측치 처리
    - customerID 열 제거
    - Churn 값을 0/1로 변환
    
    Args:
        file_path (str): CSV 파일의 경로
        
    Returns:
        pandas.DataFrame: 정제된 데이터프레임
    """
    try:
        df = pd.read_csv(file_path)
        print("✅ [data_loader] 데이터 로드 성공")
    except FileNotFoundError:
        print(f"❌ [data_loader] 오류: '{file_path}' 파일을 찾을 수 없습니다.")
        return None

    # TotalCharges가 비어있는 경우 중간값으로 채웁니다.
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # 불필요한 customerID 열을 제거합니다.
    df.drop('customerID', axis=1, inplace=True)

    # Target 변수인 Churn을 0과 1로 변환합니다.
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    print("✅ [data_loader] 데이터 기본 정제 완료")
    return df