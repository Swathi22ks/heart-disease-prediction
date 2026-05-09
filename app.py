"""
Heart Disease Risk Predictor — Streamlit Web App
Author: Swathi K S | Dataset: Framingham Heart Study
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: #f8fafc; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-left: 4px solid #e74c3c;
        margin-bottom: 10px;
    }
    .high-risk { background: linear-gradient(135deg, #ff6b6b, #ee0979);
                 color: white; padding: 20px; border-radius: 16px; text-align: center; }
    .low-risk  { background: linear-gradient(135deg, #56ab2f, #a8e063);
                 color: white; padding: 20px; border-radius: 16px; text-align: center; }
    .stButton>button {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white; border: none; border-radius: 10px;
        padding: 14px 36px; font-size: 16px; font-weight: 600;
        width: 100%; transition: transform 0.2s;
    }
    .stButton>button:hover { transform: translateY(-2px); }
    h1 { color: #2c3e50; }
</style>
""", unsafe_allow_html=True)

# ── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.dirname(__file__)
    model   = pickle.load(open(os.path.join(base, 'models/best_model.pkl'), 'rb'))
    scaler  = pickle.load(open(os.path.join(base, 'models/scaler.pkl'),     'rb'))
    imputer = pickle.load(open(os.path.join(base, 'models/imputer.pkl'),    'rb'))
    return model, scaler, imputer

model, scaler, imputer = load_models()

FEATURES = ['age','gender','currentSmoker','cigsPerDay','BPMeds',
            'prevalentStroke','prevalentHyp','diabetes','totChol',
            'sysBP','diaBP','BMI','heartRate','glucose']

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# ❤️ Heart Disease Risk Predictor")
st.markdown("**Based on the Framingham Heart Study** — Predict your 10-year coronary heart disease (CHD) risk using machine learning.")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 About This App")
    st.info("""
    This tool uses a **Logistic Regression** model trained on the Framingham Heart Study dataset
    with SMOTE balancing to handle class imbalance.

    **AUC-ROC: 0.697** — significantly better than the baseline model.

    ⚠️ *For educational purposes only. Always consult a medical professional.*
    """)
    st.markdown("---")
    st.markdown("**Dataset:** Framingham Heart Study")
    st.markdown("**Model:** Logistic Regression + SMOTE")
    st.markdown("**Features:** 14 clinical variables")
    st.markdown("---")
    st.markdown("Built by **Swathi K S** | BCA Internship Project")

# ── Input Form ────────────────────────────────────────────────────────────────
st.markdown("### 🏥 Enter Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**👤 Demographics**")
    age    = st.slider("Age (years)", 30, 80, 45)
    gender = st.radio("Gender", ["Female (0)", "Male (1)"])
    gender_val = 1 if "Male" in gender else 0

with col2:
    st.markdown("**🚬 Lifestyle**")
    currentSmoker = st.selectbox("Current Smoker?", ["No (0)", "Yes (1)"])
    smoker_val    = 1 if "Yes" in currentSmoker else 0
    cigsPerDay    = st.slider("Cigarettes per Day", 0, 60, 0)
    BPMeds        = st.selectbox("On Blood Pressure Medication?", ["No (0)", "Yes (1)"])
    bpmeds_val    = 1 if "Yes" in BPMeds else 0

with col3:
    st.markdown("**🩺 Medical History**")
    prevalentStroke = st.selectbox("History of Stroke?",        ["No (0)", "Yes (1)"])
    stroke_val      = 1 if "Yes" in prevalentStroke else 0
    prevalentHyp    = st.selectbox("History of Hypertension?",  ["No (0)", "Yes (1)"])
    hyp_val         = 1 if "Yes" in prevalentHyp else 0
    diabetes        = st.selectbox("Diabetic?",                  ["No (0)", "Yes (1)"])
    diabetes_val    = 1 if "Yes" in diabetes else 0

st.markdown("---")
st.markdown("### 📊 Clinical Measurements")

col4, col5, col6, col7 = st.columns(4)
with col4:
    totChol   = st.number_input("Total Cholesterol (mg/dL)", 100, 600, 230)
with col5:
    sysBP     = st.number_input("Systolic BP (mmHg)",        80, 300, 120)
with col6:
    diaBP     = st.number_input("Diastolic BP (mmHg)",       50, 200, 80)
with col7:
    BMI       = st.number_input("BMI", 10.0, 60.0, 25.0, step=0.1)

col8, col9 = st.columns(2)
with col8:
    heartRate = st.number_input("Heart Rate (bpm)", 40, 200, 75)
with col9:
    glucose   = st.number_input("Glucose Level (mg/dL)", 40, 400, 85)

# ── Predict ───────────────────────────────────────────────────────────────────
st.markdown("---")
if st.button("🔍 Predict CHD Risk"):
    input_data = pd.DataFrame([[
        age, gender_val, smoker_val, cigsPerDay, bpmeds_val,
        stroke_val, hyp_val, diabetes_val, totChol,
        sysBP, diaBP, BMI, heartRate, glucose
    ]], columns=FEATURES)

    input_imp = imputer.transform(input_data)
    input_sc  = scaler.transform(input_imp)
    prob      = model.predict_proba(input_sc)[0][1]
    pred      = model.predict(input_sc)[0]

    st.markdown("---")
    st.markdown("## 🎯 Prediction Result")

    r1, r2, r3 = st.columns(3)
    with r1:
        if pred == 1:
            st.markdown(f"""<div class="high-risk">
            <h2>⚠️ HIGH RISK</h2>
            <p>Elevated 10-year CHD risk detected</p>
            <h3>{prob*100:.1f}% probability</h3>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="low-risk">
            <h2>✅ LOW RISK</h2>
            <p>Lower 10-year CHD risk detected</p>
            <h3>{prob*100:.1f}% probability</h3>
            </div>""", unsafe_allow_html=True)

    with r2:
        st.markdown(f"""<div class="metric-card">
        <h4>📈 Risk Probability</h4>
        <h2>{prob*100:.1f}%</h2>
        <p>Likelihood of CHD in 10 years</p>
        </div>""", unsafe_allow_html=True)

    with r3:
        # Risk factors flagged
        flags = []
        if age > 60: flags.append("Age > 60")
        if sysBP > 140: flags.append("High Systolic BP")
        if totChol > 240: flags.append("High Cholesterol")
        if glucose > 126: flags.append("High Glucose")
        if smoker_val and cigsPerDay > 10: flags.append("Heavy Smoker")
        if BMI > 30: flags.append("Obesity (BMI>30)")
        if flags:
            st.markdown(f"""<div class="metric-card">
            <h4>🚨 Detected Risk Factors</h4>
            <ul>{"".join(f"<li>{f}</li>" for f in flags)}</ul>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="metric-card">
            <h4>✅ No Major Risk Flags</h4>
            <p>No high-risk values detected in inputs</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.info("⚕️ **Disclaimer:** This prediction is for educational purposes only and should not replace professional medical advice. Please consult a qualified healthcare provider for any health concerns.")

# ── Model Performance Section ─────────────────────────────────────────────────
with st.expander("📊 View Model Performance Metrics"):
    st.markdown("### Model Comparison Table")
    perf_data = {
        'Model':           ['Logistic Regression','Decision Tree','Random Forest','Gradient Boosting','XGBoost'],
        'Accuracy':        [0.6639, 0.7700, 0.8172, 0.8208, 0.7983],
        'AUC-ROC':         [0.6968, 0.6810, 0.6565, 0.6431, 0.5846],
        'F1-Score':        [0.3508, 0.3345, 0.1799, 0.2083, 0.1320],
        'Recall':          [0.5969, 0.3798, 0.1318, 0.1550, 0.1008],
    }
    perf_df = pd.DataFrame(perf_data).set_index('Model')
    st.dataframe(perf_df.style.highlight_max(axis=0, color='#d4edda')
                              .highlight_min(axis=0, color='#f8d7da')
                              .format('{:.4f}'))
    st.caption("✅ Best model selected by AUC-ROC (most important metric for imbalanced medical data)")
