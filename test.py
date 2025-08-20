import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# 1. 가상 데이터셋 생성
# -----------------------------
np.random.seed(42)
drugs = ["타이레놀", "아스피린", "이부프로펜", "메트포르민", "로수바스타틴"]
n = 500

data = {
    "약이름": np.random.choice(drugs, n),
    "나이": np.random.randint(10, 90, n),
    "성별": np.random.choice(["남성", "여성"], n),
    "체중": np.random.randint(40, 100, n),
    "복용량": np.random.randint(10, 500, n),
}

# 부작용 발생 확률 가상 규칙
side_effect_prob = (
    (data["나이"] - 40) * 0.01
    + (data["복용량"] / 500) * 0.3
    + np.where(pd.Series(data["약이름"]).isin(["메트포르민", "로수바스타틴"]), 0.2, 0)
    + np.random.normal(0, 0.1, n)
)
side_effect_prob = 1 / (1 + np.exp(-side_effect_prob))
side_effect = np.random.binomial(1, side_effect_prob)

df = pd.DataFrame(data)
df["부작용"] = side_effect

# -----------------------------
# 2. 데이터 전처리 & 모델 학습
# -----------------------------
X = df.drop("부작용", axis=1)
y = df["부작용"]

# 범주형 데이터 인코딩
le_drug = LabelEncoder()
le_gender = LabelEncoder()
X["약이름"] = le_drug.fit_transform(X["약이름"])
X["성별"] = le_gender.fit_transform(X["성별"])

# 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습 (Random Forest)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 3. 스트림릿 UI
# -----------------------------
st.title("💊 약물 부작용 예측 앱")
st.write("가상의 데이터셋을 기반으로 환자의 특성을 입력하면, 부작용 발생 가능성을 예측합니다.")

# 사용자 입력
drug_input = st.selectbox("약 이름", drugs)
age_input = st.slider("나이", 10, 90, 30)
gender_input = st.radio("성별", ["남성", "여성"])
weight_input = st.slider("체중 (kg)", 40, 100, 60)
dosage_input = st.slider("복용량 (mg)", 10, 500, 100)

# 입력 데이터 변환
input_data = pd.DataFrame({
    "약이름": [le_drug.transform([drug_input])[0]],
    "나이": [age_input],
    "성별": [le_gender.transform([gender_input])[0]],
    "체중": [weight_input],
    "복용량": [dosage_input]
})

# 예측
if st.button("예측하기"):
    prob = model.predict_proba(input_data)[0][1]  # 부작용 확률
    pred = model.predict(input_data)[0]

    st.subheader("📌 예측 결과")
    st.write(f"부작용 발생 확률: **{prob*100:.2f}%**")
    if pred == 1:
        st.error("⚠️ 부작용이 발생할 가능성이 있습니다.")
    else:
        st.success("✅ 부작용 발생 가능성이 낮습니다.")

    # Feature Importance 시각화
    st.subheader("📊 특성 중요도 (Feature Importance)")
    importances = model.feature_importances_
    features = X.columns
    fig, ax = plt.subplots()
    ax.barh(features, importances)
    ax.set_xlabel("중요도")
    ax.set_title("각 특성이 예측에 기여한 정도")
    st.pyplot(fig)

    # -----------------------------
    # 4. 상관관계 설명
    # -----------------------------
    st.subheader("📖 해석 및 상관관계")
    st.write("""
    - **나이**: 나이가 많을수록 부작용 발생 확률이 높아집니다.  
    - **복용량**: 복용량이 많을수록 부작용 가능성이 증가합니다.  
    - **약 이름**: 메트포르민, 로수바스타틴은 다른 약에 비해 부작용 위험도가 조금 더 높게 설정되어 있습니다.  
    - **성별, 체중**: 일부 영향을 주지만, 나이와 복용량보다는 영향력이 작습니다.  
    """)
