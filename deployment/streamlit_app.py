import streamlit as st
import pandas as pd
import pickle

# --------Load Saved Pipeline--------
with open("research/mental_health.pkl", "rb") as f:
    model = pickle.load(f)
    

st.title("Mental Health Prediction App")

# --------Numeric Feature--------
numeric_cols = { 
    'Age': (15, 75, 18)
}

# Collect numeric inputs
numeric_input = {}
for col, (min_val, max_val, default) in numeric_cols.items():
    numeric_input[col] = st.slider(col, min_value=min_val, max_value=max_val, value=default)


# --------Categorical Features--------
categorical_cols = {

    "Gender": ['Male', 'Female'],
    "self_employed": ["Yes", "No"],
    "family_history": ["Yes", "No"],
    "work_interfere": ["Never", "Rarely", "Sometimes", "Often", "Don't know"],
    "no_employees": ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"],
    "remote_work": ["Yes", "No"],
    "tech_company": ["Yes", "No"],
    "benefits": ["Yes", "No", "Don't know"],
    "care_options": ["Yes", "No", "Not sure"],
    "wellness_program": ["Yes", "No", "Don't know"],
    "seek_help": ["Yes", "No", "Don't know"],
    "anonymity": ["Yes", "No", "Don't know"],
    "leave": ["Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult", "Don't know"],
    "mental_health_consequence": ["Yes", "No", "Maybe"],
    "phys_health_consequence": ["Yes", "No", "Maybe"],
    "coworkers": ["Yes", "No", "Some of them"],
    "supervisor": ["Yes", "No", "Some of them"],
    "mental_health_interview": ["Yes", "No", "Maybe"],
    "phys_health_interview": ["Yes", "No", "Maybe"],
    "mental_vs_physical": ["Yes", "No", "Don't know"],  
    "obs_consequence": ["Yes", "No"]
}


categoric_input = {}
for col, options in categorical_cols.items():
    categoric_input[col] = st.selectbox(col, options)


# --------Create DataFrame--------
input_data = {**numeric_input, **categoric_input}
input_df = pd.DataFrame([input_data])

# predict

if st.button("Predict"):
  pred = model.predict(input_df)
  st.subheader(f"Mental Health Prediction:{'yes' if pred[0] == 1 else "No"}")
