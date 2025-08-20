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
drugs = ["DrugA", "DrugB", "DrugC", "DrugD", "DrugE"]
n = 500

data = {
    "drug_name": np.random.choice(drugs, n),
    "age": np.random.randint(10, 90, n),
    "gender": np.random.choice(["M", "F"], n),
    "weight": np.random.randint(40, 100, n),
    "dosage": np.random.randint(10, 500, n),
}

side_effect_prob = (
    (data["age"] - 40) * 0.01
    + (data["dosage"] / 500) * 0.3
    + np.where(pd.Series(data["drug_name"]).isin(["DrugC", "DrugE"]), 0.2, 0)
    + np.random.normal(0, 0.1, n)
)
side_effect_prob = 1 / (1 + np.exp(-side_effect_prob))
side_effect = np.random.binomial(1, side_effect_prob)

df = pd.DataFrame(data)
df["side_effect"] = side_effect

# -----------------------------
# 2. 데이터 전처리 & 모델 학습
# -----------------------------
X = df.drop("side_effect", axis=1)
y = df["side_effect"]

# 범주형 데이터 인코딩
le_drug = LabelEncoder()
le_gender = LabelEncoder()
X["drug_name"] = le_drug.fit_transform(X["drug_name"])
X["gender"] = le_gender.fit_transform(X["gender"])

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
gender_input = st.radio("성별", ["M", "F"])
weight_input = st.slider("체중 (kg)", 40, 100, 60)
dosage_input = st.slider("복용량 (mg)", 10, 500, 100)

# 입력 데이터 변환
input_data = pd.DataFrame({
    "drug_name": [le_drug.transform([drug_input])[0]],
    "age": [age_input],
    "gender": [le_gender.transform([gender_input])[0]],
    "weight": [weight_input],
    "dosage": [dosage_input]
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
    st.subheader("📊 Feature Importance")
    importances = model.feature_importances_
    features = X.columns
    fig, ax = plt.subplots()
    ax.barh(features, importances)
    ax.set_xlabel("Importance")
    ax.set_title("각 특성이 예측에 기여한 정도")
    st.pyplot(fig)

