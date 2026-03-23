# ================= IMPORTS =================
import streamlit as st
import pandas as pd
import pickle
import os
from tensorflow.keras.models import load_model

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Mental Health AI", page_icon="🧠", layout="wide")

# ================= PATH =================
BASE_DIR = os.path.dirname(__file__)

def load_file(path):
    return os.path.join(BASE_DIR, "..", path)

# ================= LOAD MODELS =================
tfidf = pickle.load(open(load_file("research/tfidf.pkl"), "rb"))
text_model = pickle.load(open(load_file("research/text_model.pkl"), "rb"))
structured_model = pickle.load(open(load_file("research/structured_model.pkl"), "rb"))
nn_model = load_model(load_file("research/nn_model.h5"))
nn_columns = pickle.load(open(load_file("research/nn_columns.pkl"), "rb"))

# ================= PREMIUM CSS =================
st.markdown("""
<style>
body {
    background-color: #f8fafc;
}
.block-container {
    padding-top: 2rem;
}

/* HEADER */
.header {
    text-align: center;
    padding: 20px;
}
.header h1 {
    font-size: 42px;
    margin-bottom: 5px;
}
.header p {
    color: gray;
    font-size: 18px;
}

/* CARD */
.card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.05);
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(90deg, #6366f1, #3b82f6);
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}

/* RESULT */
.result {
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    color: white;
    padding: 30px;
    border-radius: 15px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<div class="header">
    <h1>🧠 Mental Health AI Platform</h1>
    <p>AI-powered mental health risk analysis system</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ================= LAYOUT =================
left, right = st.columns([1, 1.2])

# ================= INPUT CARD =================
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📋 User Information")

    age = st.slider("Age", 15, 75, 25)
    gender = st.selectbox("Gender", ['Male', 'Female'])
    family_history = st.selectbox("Family History", ["Yes", "No"])
    work_interfere = st.selectbox("Work Stress",
                                 ["Never", "Rarely", "Sometimes", "Often", "Not sure"])

    st.subheader("🧠 Thoughts")
    comments = st.text_area("How are you feeling today?", height=120)

    predict_btn = st.button("🚀 Analyze Mental Health")

    st.markdown('</div>', unsafe_allow_html=True)

# ================= RESULT PANEL =================
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📊 Analysis Dashboard")

    if predict_btn:

        # ---------- INPUT ----------
        input_df = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "self_employed": "No",
            "family_history": family_history,
            "work_interfere": work_interfere,
            "no_employees": "1-5",
            "remote_work": "No",
            "tech_company": "Yes",
            "benefits": "Yes",
            "care_options": "Yes",
            "wellness_program": "No",
            "seek_help": "Yes",
            "anonymity": "Yes",
            "leave": "Somewhat easy",
            "mental_health_consequence": "No",
            "phys_health_consequence": "No",
            "coworkers": "Some of them",
            "supervisor": "Yes",
            "mental_health_interview": "No",
            "phys_health_interview": "No",
            "mental_vs_physical": "Yes",
            "obs_consequence": "No"
        }])

        # ---------- STRUCTURED ----------
        structured_prob = structured_model.predict_proba(input_df)[0][1]

        # ---------- TEXT ----------
        if comments.strip() == "":
            comments = "No comment"

        text_input = tfidf.transform([comments])
        text_prob = text_model.predict_proba(text_input)[0][1]

        # ---------- NEURAL NETWORK ----------
        nn_input = pd.get_dummies(input_df)

        for col in nn_columns:
            if col not in nn_input:
                nn_input[col] = 0

        nn_input = nn_input[nn_columns].astype("float32")
        nn_prob = nn_model.predict(nn_input)[0][0]

        # ---------- FINAL ----------
        final_prob = (structured_prob + text_prob + nn_prob) / 3
        percent = round(final_prob * 100, 2)

        # ================= RESULT =================
        st.markdown(f"""
        <div class="result">
            <h1>{percent}%</h1>
            <p>Mental Health Risk Score</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(float(final_prob))

        # Risk level
        if final_prob < 0.40:
            st.success("🟢 Low Risk")
        elif final_prob < 0.70:
            st.warning("🟡 Moderate Risk")
        else:
            st.error("🔴 High Risk")

        # ================= INSIGHTS =================
        st.markdown("### 🔍 Model Breakdown")

        c1, c2, c3 = st.columns(3)
        c1.metric("Structured", f"{round(structured_prob*100,2)}%")
        c2.metric("Text", f"{round(text_prob*100,2)}%")
        c3.metric("Neural", f"{round(nn_prob*100,2)}%")

    else:
        st.info("Fill details and click Analyze to see results")

    st.markdown('</div>', unsafe_allow_html=True)