import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.train import evaluate_models
from src.visualize import plot_results, plot_before_after
from src.preprocess import load_and_preprocess

# ================= UI =================
st.set_page_config(page_title="DP Stroke App", layout="wide")

st.markdown("""
<style>

/* BACKGROUND */
.stApp {
    background-color: #0f172a;
}

/* HEADINGS */
h1, h2, h3 {
    color: #38bdf8;
}

/* TEXT */
label, p, span {
    color: #e2e8f0 !important;
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(90deg, #38bdf8, #0ea5e9);
    color: black;
    border-radius: 8px;
    padding: 6px 15px;
    font-weight: 600;
}

/* INPUT */
input, select {
    background-color: #1e293b !important;
    color: white !important;
}

/* TABLE FIX */
table {
    color: white !important;
    background-color: #1e293b !important;
}
th {
    background-color: #2563eb !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

st.title("🔐 Privacy-Preserving Stroke Prediction")

data_path = "data/healthcare-dataset-stroke-data.csv"

# ================= ANALYSIS =================
st.header("📊 Privacy Analysis")

if st.button("Run Analysis"):

    baseline_acc, dp_results = evaluate_models(data_path)

    # ---------- TABLE ----------
    df = pd.DataFrame(dp_results, columns=["Epsilon","DP Accuracy"])
    df["Before Accuracy"] = baseline_acc

    df["DP Accuracy"] = df["DP Accuracy"].round(4)
    df["Before Accuracy"] = df["Before Accuracy"].round(4)

    df.columns = ["Epsilon (ε)", "After Privacy Accuracy", "Before Privacy Accuracy"]

    st.subheader("📋 Accuracy Comparison Table")
    st.table(df)   # ✅ FIXED (NO BLUR)

    # ---------- CHARTS ----------
    before = baseline_acc
    after = dp_results[2][1]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Before vs After")
        st.pyplot(plot_before_after(before, after), use_container_width=False)

    with col2:
        st.subheader("📈 Privacy vs Accuracy")
        st.pyplot(plot_results(before, dp_results), use_container_width=False)

    # ---------- INTERPRETATION ----------
    st.subheader("🧠 Interpretation")

    st.info(f"""
    Before Accuracy: {before:.4f}  
    After Accuracy (ε = 1): {after:.4f}  

    ✔ More privacy → lower accuracy  
    ✔ Less privacy → higher accuracy  

    👉 This clearly shows the privacy-utility trade-off.
    """)

# ================= PREDICTION =================
st.header("🩺 Stroke Prediction")

st.markdown("### Enter Patient Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 1, 100, 30)
    hypertension = st.selectbox("Hypertension", [0,1])
    heart = st.selectbox("Heart Disease", [0,1])

with col2:
    glucose = st.number_input("Glucose Level", 50.0, 300.0, 100.0)
    bmi = st.number_input("BMI", 10.0, 50.0, 25.0)

with col3:
    gender = st.selectbox("Gender", ["Male","Female","Other"])
    married = st.selectbox("Married", ["Yes","No"])
    work = st.selectbox("Work Type", ["Private","Self-employed","Govt_job","children","Never_worked"])
    residence = st.selectbox("Residence", ["Urban","Rural"])
    smoke = st.selectbox("Smoking", ["never smoked","formerly smoked","smokes","Unknown"])

def encode():
    gender_map = {"Female":0,"Male":1,"Other":2}
    married_map = {"No":0,"Yes":1}
    work_map = {"Govt_job":0,"children":1,"Private":2,"Self-employed":3,"Never_worked":4}
    residence_map = {"Rural":0,"Urban":1}
    smoke_map = {"Unknown":0,"formerly smoked":1,"never smoked":2,"smokes":3}

    return np.array([[
        gender_map[gender],
        age,
        hypertension,
        heart,
        married_map[married],
        work_map[work],
        residence_map[residence],
        glucose,
        bmi,
        smoke_map[smoke]
    ]])

if st.button("Predict"):

    X, y, scaler, _ = load_and_preprocess(data_path)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    user = scaler.transform(encode())

    pred = model.predict(user)[0]
    prob = model.predict_proba(user)[0][1]

    st.subheader("🔍 Result")

    if pred == 1:
        st.error(f"⚠️ High Risk ({prob*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk ({prob*100:.2f}%)")