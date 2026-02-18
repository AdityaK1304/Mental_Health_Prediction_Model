import streamlit as st
import pandas as pd
import pickle

# --------Load Saved Pipeline--------
with open("research/best_model.pkl", "rb") as f:
    best_model = pickle.load(f)

with open("research/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

st.title("🧠 Mental Health Prediction App")

# --------Numeric Feature--------
numeric_cols = { 
    'Age': (15, 75, 18)
}

numeric_input = {}
for col, (min_val, max_val, step) in numeric_cols.items():
    options = list(range(min_val, max_val + 1, step))
    numeric_input[col] = st.select_slider(col, options, value=min_val)


# --------Categorical Features--------
categorical_cols = {
    "Gender": ["Male", "Female"],
    "family_history": ["Yes", "No"],
    "work_interfere": ["Never", "Rarely", "Sometimes", "Often"],
    "Remote_work":['Yes','No'],
    "benefits": ["Yes", "No", "Don't know"],
    "care_options": ["Yes", "No", "Not sure"],
    "anonymity": ["Yes", "No", "Don't know"],
    "mental_health_consequence": ["Yes", "No", "Maybe"],
    "mental_health_interview": ["Yes", "No", "Maybe"]
}

categoric_input = {}
for col, options in categorical_cols.items():
    categoric_input[col] = st.selectbox(col, options)


# --------Create DataFrame--------
input_data = {**numeric_input, **categoric_input}
input_df = pd.DataFrame([input_data])

# predict

if st.button("Predict"):
  pred = best_model.predict(input_df)
  st.subheader(f"Mental Health Prediction:{'yes' if pred[0] == 1 else "No"}")
