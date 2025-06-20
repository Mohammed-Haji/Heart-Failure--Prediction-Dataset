
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# --- Load The Models and Scaler ---
# We use a cache to prevent reloading the model on every interaction
@st.cache_resource
def load_assets():
    """Loads the trained model, scaler, and training columns."""
    try:
        with open('models/heart_disease_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('processed_data/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('processed_data/train_cols.pkl', 'rb') as f:
            train_cols = pickle.load(f)
        return model, scaler, train_cols
    except FileNotFoundError:
        st.error("Error: Model or data files not found. Please ensure 'train_model.py' and 'preprocess.py' have been run.")
        return None, None, None

model, scaler, train_cols = load_assets()

# --- App Title and Description ---
st.title("Heart Disease Prediction App ❤️")
st.markdown("This app uses a machine learning model to predict the likelihood of a patient having heart disease based on their medical attributes.")
st.markdown("---")

# --- Sidebar for User Input ---
st.sidebar.header("Patient Input Features")

def user_input_features():
    """Creates sidebar widgets and returns user input as a dictionary."""
    age = st.sidebar.slider('Age', 20, 90, 50)
    sex = st.sidebar.selectbox('Sex', ('M', 'F'))
    chest_pain_type = st.sidebar.selectbox('Chest Pain Type', ('ATA', 'NAP', 'ASY', 'TA'))
    resting_bp = st.sidebar.slider('Resting Blood Pressure (mm Hg)', 80, 200, 120)
    cholesterol = st.sidebar.slider('Cholesterol (mg/dl)', 100, 600, 200)
    fasting_bs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', (0, 1))
    resting_ecg = st.sidebar.selectbox('Resting ECG', ('Normal', 'ST', 'LVH'))
    max_hr = st.sidebar.slider('Maximum Heart Rate', 60, 220, 150)
    exercise_angina = st.sidebar.selectbox('Exercise-Induced Angina', ('N', 'Y'))
    oldpeak = st.sidebar.slider('Oldpeak (ST depression)', 0.0, 6.5, 1.0)
    st_slope = st.sidebar.selectbox('ST Slope', ('Up', 'Flat', 'Down'))

    data = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain_type,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }
    return data

input_data = user_input_features()

# --- Main Page: Display Prediction ---
if model and scaler and train_cols is not None:
    st.header("Prediction")

    # Convert user input into a DataFrame
    input_df = pd.DataFrame([input_data])

    # One-hot encode the categorical features
    input_df = pd.get_dummies(input_df, drop_first=True)

    # Align the input DataFrame columns with the training columns
    input_df = input_df.reindex(columns=train_cols, fill_value=0)

    # Scale the user input
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error("This patient is likely to have Heart Disease.")
    else:
        st.success("This patient is likely to NOT have Heart Disease.")

    st.subheader("Prediction Probability:")
    st.write(f"Confidence (No Heart Disease): {probability[0][0]:.2%}")
    st.write(f"Confidence (Heart Disease): {probability[0][1]:.2%}")

    # Display the user input for confirmation
    st.markdown("---")
    st.header("User Input Summary")
    st.write(pd.DataFrame([input_data]).T.rename(columns={0: 'Values'}))

else:
    st.warning("Please run the training scripts to generate the necessary model files.")
