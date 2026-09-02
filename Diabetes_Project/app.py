# Import Streamlit for creating the web application
import streamlit as st

# Import NumPy for creating the input data
import numpy as np

# Import joblib for loading the saved model and scaler
import joblib

# Import os for handling file paths
import os


# Get the folder where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Load the trained Logistic Regression model
model_path = os.path.join(
    BASE_DIR,
    "logistic_regression_model.pkl"
)

model = joblib.load(model_path)


# Load the StandardScaler
scaler_path = os.path.join(
    BASE_DIR,
    "scaler.pkl"
)

scaler = joblib.load(scaler_path)


# Application title
st.title("Diabetes Prediction App")

# Application description
st.write(
    "Enter the patient details below to predict the diabetes outcome."
)


# Input: Pregnancies
pregnancies = st.number_input(
    "Pregnancies",
    min_value=0,
    value=1
)


# Input: Glucose
glucose = st.number_input(
    "Glucose",
    min_value=0.0,
    value=120.0
)


# Input: Blood Pressure
blood_pressure = st.number_input(
    "Blood Pressure",
    min_value=0.0,
    value=70.0
)


# Input: Skin Thickness
skin_thickness = st.number_input(
    "Skin Thickness",
    min_value=0.0,
    value=20.0
)


# Input: Insulin
insulin = st.number_input(
    "Insulin",
    min_value=0.0,
    value=80.0
)


# Input: BMI
bmi = st.number_input(
    "BMI",
    min_value=0.0,
    value=25.0
)


# Input: Diabetes Pedigree Function
diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    value=0.5
)


# Input: Age
age = st.number_input(
    "Age",
    min_value=0,
    value=30
)


# Create the prediction button
if st.button("Predict"):

    # Store user inputs in the same order
    # as the features used during model training
    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]])


    # Standardize the input using the saved scaler
    input_scaled = scaler.transform(input_data)


    # Make the prediction
    prediction = model.predict(input_scaled)


    # Calculate probability of diabetes
    probability = model.predict_proba(input_scaled)[0][1]


    # Display the prediction
    if prediction[0] == 1:
        st.error("Prediction: Diabetes")
    else:
        st.success("Prediction: No Diabetes")


    # Display probability
    st.write(
        f"Probability of Diabetes: {probability:.2%}"
    )