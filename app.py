"""
Diabetes Risk Prediction App
Model : XGBoost Classifier (9 features)
Scaler: StandardScaler fitted ONLY on
        [age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level]
        gender_Male / gender_Other / smoking_history stay UNSCALED
        (verified against scaler.mean_ / scaler.feature_names_in_)
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

MODEL_PATH = "xgb_model.pkl"
SCALER_PATH = "scaler.pkl"

# Columns the scaler was fit on (continuous / binary numeric features only)
SCALE_COLS = ["age", "hypertension", "heart_disease", "bmi", "HbA1c_level", "blood_glucose_level"]

# Exact column order the XGBoost model expects
FEATURE_ORDER = SCALE_COLS + ["gender_Male", "gender_Other", "smoking_history"]

# smoking_history label-encoding used at training time (alphabetical order)
SMOKING_MAP = {"No Info": 0, "current": 1, "ever": 2, "former": 3, "never": 4, "not current": 5}


# ----------------------------------------------------------------------------
# Load artifacts (cached so files load once per session)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        return None, None, f"'{MODEL_PATH}' not found next to app.py."
    if not os.path.exists(SCALER_PATH):
        return None, None, f"'{SCALER_PATH}' not found next to app.py."
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler, None


model, scaler, load_error = load_artifacts()

st.title("🩺 Diabetes Risk Prediction")
st.caption("XGBoost classifier trained on the Kaggle Diabetes Prediction dataset")

if load_error:
    st.error(load_error)
    st.stop()

st.success("Model & scaler loaded successfully ✅")


# ----------------------------------------------------------------------------
# Core prediction function
# ----------------------------------------------------------------------------
def predict(age, hypertension, heart_disease, bmi, hba1c, glucose, gender, smoking):
    gender_Male = 1.0 if gender == "Male" else 0.0
    gender_Other = 1.0 if gender == "Other" else 0.0
    smoking_val = SMOKING_MAP[smoking]

    # 1. Scale ONLY the columns the scaler was fit on
    to_scale = pd.DataFrame(
        [[age, hypertension, heart_disease, bmi, hba1c, glucose]], columns=SCALE_COLS
    )
    scaled = scaler.transform(to_scale)
    scaled_df = pd.DataFrame(scaled, columns=SCALE_COLS)

    # 2. Append unscaled encoded features, in the exact order the model expects
    scaled_df["gender_Male"] = gender_Male
    scaled_df["gender_Other"] = gender_Other
    scaled_df["smoking_history"] = smoking_val
    final_row = scaled_df[FEATURE_ORDER]

    pred = int(model.predict(final_row)[0])
    prob = model.predict_proba(final_row)[0]
    return pred, prob, final_row


# ----------------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------------
with st.form("prediction_form"):
    st.subheader("Patient Information")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0.0, max_value=120.0, value=35.0, step=1.0)
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
        hba1c = st.number_input("HbA1c Level (%)", min_value=3.0, max_value=15.0, value=5.5, step=0.1)
        glucose = st.number_input("Blood Glucose Level (mg/dL)", min_value=50, max_value=400, value=120, step=1)

    with col2:
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
        smoking = st.selectbox("Smoking History", list(SMOKING_MAP.keys()))

    submitted = st.form_submit_button("Predict", use_container_width=True)


# ----------------------------------------------------------------------------
# Prediction output
# ----------------------------------------------------------------------------
if submitted:
    pred, prob, final_row = predict(
        age,
        1 if hypertension == "Yes" else 0,
        1 if heart_disease == "Yes" else 0,
        bmi, hba1c, glucose, gender, smoking,
    )

    st.divider()
    st.subheader("Result")

    if pred == 1:
        st.error(f"⚠️ High Risk of Diabetes — Confidence: {prob[1]*100:.2f}%")
    else:
        st.success(f"✅ Low Risk of Diabetes — Confidence: {prob[0]*100:.2f}%")

    c1, c2 = st.columns(2)
    c1.metric("P(Non-Diabetic)", f"{prob[0]*100:.2f}%")
    c2.metric("P(Diabetic)", f"{prob[1]*100:.2f}%")

    with st.expander("See model input (after scaling)"):
        st.dataframe(final_row, use_container_width=True)

    st.caption("Educational tool only — not a substitute for professional medical diagnosis.")


# ----------------------------------------------------------------------------
# Sidebar sanity check
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Model Sanity Check")
    st.write("Confirms the pipeline behaves logically before you trust it.")
    if st.button("Run Sanity Check"):
        cases = {
            "Healthy young": (20, 0, 0, 21.0, 4.5, 90, "Female", "never"),
            "Unhealthy senior": (65, 1, 1, 38.0, 9.5, 260, "Male", "current"),
        }
        for label, (a, h, hd, b, hb, g, gen, smk) in cases.items():
            p, pr, _ = predict(a, h, hd, b, hb, g, gen, smk)
            st.write(f"**{label}** → {'Diabetic' if p==1 else 'Non-Diabetic'} "
                     f"(P(diabetic)={pr[1]*100:.1f}%)")
