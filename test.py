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
drugs = ["Tylenol", "Aspirin", "Ibuprofen", "Metformin", "Rosuvastatin"]
n = 500

data = {
    "Drug": np.random.choice(drugs, n),
    "Age": np.random.randint(10, 90, n),
    "Gender": np.random.choice(["Male", "Female"], n),
    "Weight": np.random.randint(40, 100, n),
    "Dosage": np.random.randint(10, 500, n),
}

# 부작용 발생 확률 가상 규칙
side_effect_prob = (
    (data["Age"] - 40) * 0.01
    + (data["Dosage"] / 500) * 0.3
    + np.where(pd.Series(data["Drug"]).isin(["Metformin", "Rosuvastatin"]), 0.2, 0)
    + np.random.normal(0, 0.1, n)
)
side_effect_prob = 1 / (1 + np.exp(-side_effect_prob))
side_effect = np.random.binomial(1, side_effect_prob)

df = pd.DataFrame(data)
df["SideEffect"] = side_effect

# -----------------------------
# 2. 데이터 전처리 & 모델 학습
# -----------------------------
X = df.drop("SideEffect", axis=1)
y = df["SideEffect"]

# 범주형 인코딩
le_drug = LabelEncoder()
le_gender = LabelEncoder()
X["Drug"] = le_drug.fit_transform(X["Drug"])
X["Gender"] = le_gender.fit_transform(X["Gender"])

# 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest 모델
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 3. 스트림릿 UI
# -----------------------------
st.title("💊 Drug Side Effect Prediction App")
st.write("Input patient characteristics to predict the probability of side effects.")

# 사용자 입력
drug_input = st.selectbox("Drug", drugs)
age_input = st.slider("Age", 10, 90, 30)
gender_input = st.radio("Gender", ["Male", "Female"])
weight_input = st.slider("Weight (kg)", 40, 100, 60)
dosage_input = st.slider("Dosage (mg)", 10, 500, 100)

# 입력 데이터 변환
input_data = pd.DataFrame({
    "Drug": [le_drug.transform([drug_input])[0]],
    "Age": [age_input],
    "Gender": [le_gender.transform([gender_input])[0]],
    "Weight": [weight_input],
    "Dosage": [dosage_input]
})

# -----------------------------
# 4. 예측
# -----------------------------
if st.button("Predict"):
    prob = model.predict_proba(input_data)[0][1]
    pred = model.predict(input_data)[0]

    st.subheader("📌 Prediction Result")
    st.write(f"Probability of side effect: **{prob*100:.2f}%**")
    if pred == 1:
        st.error("⚠️ There is a risk of side effect.")
    else:
        st.success("✅ Low risk of side effect.")

    # -----------------------------
    # 5. Feature Importance 그래프
    # -----------------------------
    st.subheader("📊 Feature Importance")
    importances = model.feature_importances_
    features = X.columns
    fig, ax = plt.subplots()
    ax.barh(features, importances)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Features")
    ax.set_title("Feature Importance in Prediction")
    st.pyplot(fig)

    # -----------------------------
    # 6. 입력값 분포 시각화
    # -----------------------------
    st.subheader("📈 Input Value vs Dataset Distribution")
    for col in ["Age", "Weight", "Dosage"]:
        fig, ax = plt.subplots()
        ax.hist(df[col], bins=20, alpha=0.7, label="Dataset")
        ax.axvline(input_data[col][0], color='r', linestyle='dashed', linewidth=2, label="Input")
        ax.set_title(f"{col} Distribution")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.legend()
        st.pyplot(fig)

    # -----------------------------
    # 7. 상호작용 효과 기반 자동 해석
    # -----------------------------
    st.subheader("📖 Risk Factor Interpretation")
    interpretations = []

    # 나이, 복용량, 약 이름 상호작용
    if drug_input in ["Metformin", "Rosuvastatin"] and age_input > 60 and dosage_input > 300:
        interpretations.append("High age + high dosage + this drug → significantly higher risk.")
    else:
        if age_input > 60:
            interpretations.append("Age is high → increases risk.")
        elif age_input < 20:
            interpretations.append("Age is low → lower risk.")
        else:
            interpretations.append("Age is moderate → moderate risk.")

        if dosage_input > 300:
            interpretations.append("High dosage → increases risk.")
        else:
            interpretations.append("Dosage is moderate/low → lower risk.")

        if drug_input in ["Metformin", "Rosuvastatin"]:
            interpretations.append(f"{drug_input} → slightly higher risk than other drugs.")
        else:
            interpretations.append(f"{drug_input} → normal risk level.")

    # 성별, 체중
    if gender_input == "Female":
        interpretations.append("Female → minor increase in risk.")
    else:
        interpretations.append("Male → normal risk.")

    if weight_input > 80:
        interpretations.append("High weight → minor impact on risk.")
    else:
        interpretations.append("Weight is normal → minor impact.")

    for line in interpretations:
        st.write(f"- {line}")
