import streamlit as st
import pandas as pd
import pickle

# Load trained pipeline
with open("research/mental_health.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🧠 Mental Health Treatment Prediction")

# Numeric
age = st.slider("Age", 15, 75, 25)

# Categorical
gender = st.selectbox("Gender", ['Male', 'Female'])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
family_history = st.selectbox("Family History", ["Yes", "No"])
work_interfere = st.selectbox("Work Interfere", ["Never", "Rarely", "Sometimes", "Often", "Not sure"])

no_employees = st.selectbox("No Employees", ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
remote_work = st.selectbox("Remote Work", ["Yes", "No"])
tech_company = st.selectbox("Tech Company", ["Yes", "No"])
benefits = st.selectbox("Benefits", ["Yes", "No", "Don't know"])
care_options = st.selectbox("Care Options", ["Yes", "No", "Not sure"])
wellness_program = st.selectbox("Wellness Program", ["Yes", "No", "Don't know"])
seek_help = st.selectbox("Seek Help", ["Yes", "No", "Don't know"])
anonymity = st.selectbox("Anonymity", ["Yes", "No", "Don't know"])
leave = st.selectbox("Leave", ["Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult", "Don't know"])
mental_health_consequence = st.selectbox("Mental Health Consequence", ["Yes", "No", "Maybe"])
phys_health_consequence = st.selectbox("Physical Health Consequence", ["Yes", "No", "Maybe"])
coworkers = st.selectbox("Coworkers Support", ["Yes", "No", "Some of them"])
supervisor = st.selectbox("Supervisor Support", ["Yes", "No", "Some of them"])
mental_health_interview = st.selectbox("Mental Health Interview", ["Yes", "No", "Maybe"])
phys_health_interview = st.selectbox("Physical Health Interview", ["Yes", "No", "Maybe"])
mental_vs_physical = st.selectbox("Mental vs Physical", ["Yes", "No", "Don't know"])
obs_consequence = st.selectbox("Observed Consequence", ["Yes", "No"])

if st.button("Predict"):

    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "self_employed": self_employed,
        "family_history": family_history,
        "work_interfere": work_interfere,
        "no_employees": no_employees,
        "remote_work": remote_work,
        "tech_company": tech_company,
        "benefits": benefits,
        "care_options": care_options,
        "wellness_program": wellness_program,
        "seek_help": seek_help,
        "anonymity": anonymity,
        "leave": leave,
        "mental_health_consequence": mental_health_consequence,
        "phys_health_consequence": phys_health_consequence,
        "coworkers": coworkers,
        "supervisor": supervisor,
        "mental_health_interview": mental_health_interview,
        "phys_health_interview": phys_health_interview,
        "mental_vs_physical": mental_vs_physical,
        "obs_consequence": obs_consequence
    }])

    # Get probability of treatment (class 1)
    treatment_prob = model.predict_proba(input_data)[0][1]
    percent = round(treatment_prob * 100, 2)

    st.subheader("Mental Health Assessment Result")
    st.write(f"🧠 Your mental health treatment likelihood is **{percent}%**.")
    st.progress(treatment_prob)

    # Logical thresholding
    if treatment_prob < 0.40:
        st.success("🟢 Low Risk Level")
        st.write("Your responses indicate a low likelihood of requiring professional treatment.")
    
    elif 0.40 <= treatment_prob < 0.70:
        st.warning("🟡 Moderate Risk Level")
        st.write("There are moderate indicators present. Monitoring your mental well-being or consulting a professional may be beneficial.")
    
    else:
        st.error("🔴 High Risk Level")
        st.write("Your responses strongly suggest that professional mental health support may be beneficial.")

    st.caption("⚠️ This result is based on a machine learning prediction and should not be considered a medical diagnosis.")