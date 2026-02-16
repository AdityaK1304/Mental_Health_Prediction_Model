import streamlit as st
import pandas as pd
import pickle

with open('research/best_model.pkl', 'rb') as file:
    best_model = pickle.load(file)

with open('research/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

    # numerical features
numeric_cols = {
        'Age':(3,75,1)
    }

    # categorical features
    # -------- Categorical Columns --------
categorical_cols = {
    "Gender": ["Male", "Female", "Other"],
    "family_history": ["Yes", "No"],
    "work_interfere": ["Never", "Rarely", "Sometimes", "Often"],
    "benefits": ["Yes", "No", "Don't know"],
    "care_options": ["Yes", "No", "Not sure"],
    "anonymity": ["Yes", "No", "Don't know"],
    "leave": ["Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult", "Don't know"],
    "mental_health_consequence": ["Yes", "No", "Maybe"],
    "coworkers": ["Yes", "No", "Some of them"],
    "supervisor": ["Yes", "No", "Some of them"],
    "mental_health_interview": ["Yes", "No", "Maybe"],
    "obs_consequence": ["Yes", "No"]
}

# collect numerical inputs

numeric_input = {}
for col, (min_val, max_val, step) in numeric_cols.items():
    # Generate a list of options for the selectbox
    options = list(range(min_val, max_val + 1, step))
    numeric_input[col] = st.selectbox(col, options, index=options.index(min_val))

# collect categoric input

categoric_input = {}
for col, options in categorical_cols.items():
  categoric_input[col] = st.selectbox(col, options)

# combine input into DataFrame

input_data = {**numeric_input,**categoric_input}
input_df = pd.DataFrame([input_data])

# predict

if st.button("Predict"):
  pred = best_model.predict(input_df)
  st.subheader(f"Mental Health Prediction:{'yes' if pred[0] == 1 else "No"}")
