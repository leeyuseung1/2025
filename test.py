import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정 (Windows: 맑은 고딕, Mac: AppleGothic, Linux: 나눔고딕)
plt.rcParams['font.family'] = 'Malgun Gothic'   # 윈도우
# plt.rcParams['font.family'] = 'AppleGothic'   # 맥
# plt.rcParams['font.family'] = 'NanumGothic'   # 리눅스

# 음수 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False
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

# 범주형 인코딩
le_drug = LabelEncoder()
le_gender = LabelEncoder()
X["약이름"] = le_drug.fit_transform(X["약이름"])
X["성별"] = le_gender.fit_transform(X["성별"])

# 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest 모델
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 3. 스트림릿 UI
# -----------------------------
st.title("💊 약물 부작용 예측 앱")
st.write("환자의 특성을 입력하면 부작용 발생 확률을 예측합니다.")

# 사용자 입력
drug_input = st.selectbox("약 이름", drugs)
age_input = st.slider("나이", 10, 90, 30)
gender_input = st.radio("성별", ["남성", "여성"])
weight_input = st.slider("체중(kg)", 40, 100, 60)
dosage_input = st.slider("복용량(mg)", 10, 500, 100)

# 입력 데이터 변환
input_data = pd.DataFrame({
    "약이름": [le_drug.transform([drug_input])[0]],
    "나이": [age_input],
    "성별": [le_gender.transform([gender_input])[0]],
    "체중": [weight_input],
    "복용량": [dosage_input]
})

# -----------------------------
# 4. 예측
# -----------------------------
if st.button("예측하기"):
    prob = model.predict_proba(input_data)[0][1]
    pred = model.predict(input_data)[0]

    st.subheader("📌 예측 결과")
    st.write(f"부작용 발생 확률: **{prob*100:.2f}%**")
    if pred == 1:
        st.error("⚠️ 부작용이 발생할 가능성이 있습니다.")
    else:
        st.success("✅ 부작용 발생 가능성이 낮습니다.")

    # -----------------------------
    # 5. Feature Importance 그래프
    # -----------------------------
    st.subheader("📊 특성 중요도")
    importances = model.feature_importances_
    features = X.columns
    fig, ax = plt.subplots()
    ax.barh(features, importances)
    ax.set_xlabel("중요도")
    ax.set_ylabel("특성")
    ax.set_title("예측에 기여한 특성 중요도")
    st.pyplot(fig)

    # -----------------------------
    # 6. 입력값 분포 시각화
    # -----------------------------
    st.subheader("📈 입력값과 전체 데이터 분포 비교")
    for col in ["나이", "체중", "복용량"]:
        fig, ax = plt.subplots()
        ax.hist(df[col], bins=20, alpha=0.7, label="전체 데이터")
        ax.axvline(input_data[col][0], color='r', linestyle='dashed', linewidth=2, label="입력값")
        ax.set_title(f"{col} 분포")
        ax.set_xlabel(col)
        ax.set_ylabel("빈도")
        ax.legend()
        st.pyplot(fig)

    # -----------------------------
    # 7. 상호작용 효과 기반 자동 해석
    # -----------------------------
    st.subheader("📖 위험 요인 해석")
    interpretations = []

    # 나이, 복용량, 약 이름 상호작용
    if drug_input in ["메트포르민", "로수바스타틴"] and age_input > 60 and dosage_input > 300:
        interpretations.append("고연령 + 고용량 + 해당 약물 → 부작용 위험이 크게 증가합니다.")
    else:
        if age_input > 60:
            interpretations.append("나이가 많음 → 부작용 위험 증가")
        elif age_input < 20:
            interpretations.append("나이가 적음 → 부작용 위험 낮음")
        else:
            interpretations.append("나이가 보통 → 부작용 위험 보통")

        if dosage_input > 300:
            interpretations.append("복용량이 많음 → 부작용 위험 증가")
        else:
            interpretations.append("복용량 보통/적음 → 부작용 위험 낮음")

        if drug_input in ["메트포르민", "로수바스타틴"]:
            interpretations.append(f"{drug_input} → 다른 약물보다 위험이 조금 높음")
        else:
            interpretations.append(f"{drug_input} → 일반적인 위험 수준")

    # 성별, 체중
    if gender_input == "여성":
        interpretations.append("여성 → 부작용 위험 약간 증가")
    else:
        interpretations.append("남성 → 일반적인 위험 수준")

    if weight_input > 80:
        interpretations.append("체중이 높음 → 부작용 위험에 약간 영향")
    else:
        interpretations.append("체중 보통 → 위험에 약간 영향")

    for line in interpretations:
        st.write(f"- {line}")
