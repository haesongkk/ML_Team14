import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os # 주석: 폴더를 만들거나 경로를 다루기 위해 'os' 라이브러리를 불러옵니다.

# --------------------------------------------------------------------------
# 0. 경로 설정 및 데이터 불러오기 (개선된 방식)
# --------------------------------------------------------------------------

# 주석: [개선점 1] 상대 경로 사용
# 이 코드를 프로젝트 최상위 폴더('customer-churn-prediction')에서 실행하면,
# 아래 경로는 항상 올바르게 동작합니다.
file_path = 'data/01_raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'

# 주석: [개선점 2] 그래프를 저장할 폴더 경로를 변수로 만듭니다.
# reports/figures 폴더는 처음에 제안해주신 구조에 있던 폴더입니다.
save_dir = 'reports/figures'

# 주석: 저장할 폴더가 없으면 자동으로 생성합니다. exist_ok=True는 폴더가 이미 있어도 오류를 내지 않습니다.
os.makedirs(save_dir, exist_ok=True)

try:
    df = pd.read_csv(file_path)
    print(f"✅ 데이터 로드 성공: {file_path}")
except FileNotFoundError:
    print(f"❌ 파일 경로를 찾을 수 없습니다: {file_path}")
    print("스크립트를 프로젝트 최상위 폴더에서 실행했는지 확인해주세요.")
    # 파일이 없으면 더 이상 진행할 수 없으므로 스크립트를 중단합니다.
    exit()

# --------------------------------------------------------------------------
# 1. 이탈 현황 수치 및 비율 계산
# --------------------------------------------------------------------------

# 이탈 여부('Churn')에 따른 고객 수를 계산합니다.
churn_counts = df['Churn'].value_counts()

# 이탈 여부('Churn')에 따른 고객 비율을 백분율로 계산합니다.
churn_proportions = df['Churn'].value_counts(normalize=True) * 100

# 계산된 결과를 보기 쉽게 출력합니다.
print("\n--- 고객 이탈 현황 (정확한 수치 및 비율) ---")
print("\n[이탈 여부별 고객 수]")
print(f"유지 (No):  {churn_counts['No']} 명")
print(f"이탈 (Yes): {churn_counts['Yes']} 명")

print("\n[이탈 여부별 고객 비율]")
print(f"유지 (No):  {churn_proportions['No']:.2f}%")
print(f"이탈 (Yes): {churn_proportions['Yes']:.2f}%")
print("----------------------------------------\n")

# 데이터 정제
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)


# --------------------------------------------------------------------------
# 2. 데이터 시각화 (그래프 분리 및 파일 저장)
# --------------------------------------------------------------------------
sns.set(style="whitegrid")

# 각 그래프에 대한 파일 이름을 미리 정의합니다.
plot_filenames = {
    "churn_distribution": "01_churn_distribution.png",
    "churn_by_contract": "02_churn_by_contract.png",
    "churn_by_internet_service": "03_churn_by_internet_service.png",
    "churn_by_payment_method": "04_churn_by_payment_method.png",
    "monthly_charges_dist": "05_monthly_charges_dist.png",
    "tenure_dist": "06_tenure_dist.png"
}

def save_and_show_plot(fig_name):
    """그래프를 파일로 저장하고 화면에 보여주는 함수"""
    file_save_path = os.path.join(save_dir, fig_name)
    plt.savefig(file_save_path, dpi=300, bbox_inches='tight') # bbox_inches='tight'는 잘림 방지
    print(f"✅ 그래프 저장 완료: {file_save_path}")
    plt.show()

# (1) Churn Distribution -> 이탈 분포 : yes는 이탈한 고객, no는 유지한 고객 -> 데이터 불균형 확인
plt.figure(figsize=(8, 5))
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution (Target)', fontsize=15)
save_and_show_plot(plot_filenames["churn_distribution"])

# (2) Churn by Contract Type -> 계약 유형별 이탈  -> 계약 기간이 짧을수록 고객이 서비스를 쉽게 해지
plt.figure(figsize=(8, 5))
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title('Churn by Contract Type', fontsize=15)
save_and_show_plot(plot_filenames["churn_by_contract"])

# (3) Churn by Internet Service -> 인터넷 서비스 종류별 이탈 -> 고객의 이탈률: 광랜 서비스 > DSL 이용 고객
plt.figure(figsize=(8, 5))
sns.countplot(x='InternetService', hue='Churn', data=df)
plt.title('Churn by Internet Service', fontsize=15)
save_and_show_plot(plot_filenames["churn_by_internet_service"])

# (4) Churn by Payment Method -> 결제 방법별 이탈 -> 전자 청구서를 사용하는 고객의 이탈률이 더 높음
plt.figure(figsize=(10, 6))
order = df['PaymentMethod'].value_counts().index
sns.countplot(x='PaymentMethod', hue='Churn', data=df, order=order)
plt.title('Churn by Payment Method', fontsize=15)
plt.xticks(rotation=30, ha='right')
save_and_show_plot(plot_filenames["churn_by_payment_method"])

# (5) Distribution of Monthly Charges -> 월별 요금 분포 -> 월별 요금이 높은 고객이 이탈할 가능성이 더 높음
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='MonthlyCharges', hue='Churn', kde=True, multiple="stack")
plt.title('Distribution of Monthly Charges by Churn', fontsize=15)
save_and_show_plot(plot_filenames["monthly_charges_dist"])

# (6) Distribution of Tenure -> 고객 유지 기간 분포 -> 서비스 사용 기간이 짧은 초기(0~10개월) 고객이 이탈할 가능성이 더 높음
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='tenure', hue='Churn', kde=True, multiple="stack")
plt.title('Distribution of Tenure by Churn', fontsize=15)
save_and_show_plot(plot_filenames["tenure_dist"])