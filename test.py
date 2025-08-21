import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib

# -----------------------------
# 0. Font setting (English only)
# -----------------------------
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# -----------------------------
# 1. Create Synthetic Dataset
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

# Side effect probability (virtual rule)
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
# 2. Data Preprocessing & Model Training
# -----------------------------
X = df.drop("SideEffect", axis=1)
y = df["SideEffect"]

# Label Encoding
le_drug = LabelEncoder()
le_gender = LabelEncoder()
X["Drug"] = le_drug.fit_transform(X["Drug"])
X["Gender"] = le_gender.fit_transform(X["Gender"])

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 3. Streamlit UI
# -----------------------------
st.title("💊 Drug Side Effect Prediction App")
st.write("Enter patient characteristics to predict the probability of side effects.")

# User Input
drug_input = st.selectbox("Drug", drugs)
age_input = st.slider("Age", 10, 90, 30)
gender_input = st.radio("Gender", ["Male", "Female"])
weight_input = st.slider("Weight (kg)", 40, 100, 60)
dosage_input = st.slider("Dosage (mg)", 10, 500, 100)

# Input Transformation
input_data = pd.DataFrame({
    "Drug": [le_drug.transform([drug_input])[0]],
    "Age": [age_input],
    "Gender": [le_gender.transform([gender_input])[0]],
    "Weight": [weight_input],
    "Dosage": [dosage_input]
})

# -----------------------------
# 4. Prediction
# -----------------------------
if st.button("Predict"):
    prob = model.predict_proba(input_data)[0][1]
    pred = model.predict(input_data)[0]

    st.subheader("📌 Prediction Result")
    st.write(f"Probability of side effect: **{prob*100:.2f}%**")
    if pred == 1:
        st.error("⚠️ High risk of side effect.")
    else:
        st.success("✅ Low risk of side effect.")

    # -----------------------------
    # 5. Feature Importance Graph
    # -----------------------------
    st.subheader("📊 Feature Importance")
    importances = model.feature_importances_
    features = X.columns
    fig, ax = plt.subplots()
    ax.barh(features, importances)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Importances")
    st.pyplot(fig)

    # -----------------------------
    # 6. Input Distribution Visualization
    # -----------------------------
    st.subheader("📈 Input vs Overall Distribution")
    for col in ["Age", "Weight", "Dosage"]:
        fig, ax = plt.subplots()
        ax.hist(df[col], bins=20, alpha=0.7, label="Overall Data")
        ax.axvline(input_data[col][0], color='r', linestyle='dashed', linewidth=2, label="Input Value")
        ax.set_title(f"{col} Distribution")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)

    # -----------------------------
    # 7. Automated Risk Interpretation
    # -----------------------------
    st.subheader("📖 Risk Factor Interpretation")
    interpretations = []

    # Interaction: Drug, Age, Dosage
    if drug_input in ["Metformin", "Rosuvastatin"] and age_input > 60 and dosage_input > 300:
        interpretations.append("High age + high dosage + selected drug → Risk significantly increases.")
    else:
        if age_input > 60:
            interpretations.append("Older age → Higher risk of side effect")
        elif age_input < 20:
            interpretations.append("Younger age → Lower risk of side effect")
        else:
            interpretations.append("Moderate age → Average risk")

        if dosage_input > 300:
            interpretations.append("High dosage → Higher risk")
        else:
            interpretations.append("Moderate/low dosage → Lower risk")

        if drug_input in ["Metformin", "Rosuvastatin"]:
            interpretations.append(f"{drug_input} → Slightly higher risk than other drugs")
        else:
            interpretations.append(f"{drug_input} → Normal risk level")

    # Gender, Weight
    if gender_input == "Female":
        interpretations.append("Female → Slightly higher risk")
    else:
        interpretations.append("Male → Normal risk level")

    if weight_input > 80:
        interpretations.append("High weight → May slightly influence risk")
    else:
        interpretations.append("Normal weight → Little influence on risk")

    for line in interpretations:
        st.write(f"- {line}")
