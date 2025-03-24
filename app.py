import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib  # To load the scaler

# Load the trained model
model = load_model("severity_ann_model.h5")

# Load the scaler used for preprocessing
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.title("ANN Model Prediction App")
st.write("Enter the input values to predict the output.")

# Input fields for user
val1 = st.number_input("Enter value for Feature 1")
val2 = st.number_input("Enter value for Feature 2")
val3 = st.number_input("Enter value for Feature 3")

# Convert input to NumPy array and scale
input_data = np.array([[val1, val2, val3]])
input_data_scaled = scaler.transform(input_data)

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    predicted_class = np.argmax(prediction)
    st.write(f"Predicted Class: {predicted_class}")
