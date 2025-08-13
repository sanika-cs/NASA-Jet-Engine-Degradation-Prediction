# frond.py
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

import joblib

# ----------------------------
# Load your trained model & scaler
# ----------------------------
from tensorflow.keras.losses import MeanSquaredError

model = load_model(
    r"C:\Users\sanik\Downloads\lstm_rul_model.h5",
    custom_objects={"mse": MeanSquaredError()}
)

scaler = joblib.load(r"C:\Users\sanik\scaler.pkl")

# ----------------------------
# App title
# ----------------------------
st.title("NASA Jet Engine Degradation Prediction")
st.write("Enter sensor readings to predict Remaining Useful Life (RUL)")

# ----------------------------
# Input fields for user
# ----------------------------
op1 = st.number_input("Operational Setting 1", value=0.0)
op2 = st.number_input("Operational Setting 2", value=0.0)
op3 = st.number_input("Operational Setting 3", value=0.0)

sensor_inputs = []
for i in range(1, 22):  # Example: 21 sensors
    sensor_inputs.append(
        st.number_input(f"Sensor {i}", value=0.0)
    )

# ----------------------------
# Prediction button
# ----------------------------
if st.button("Predict RUL"):
    # Prepare input
    input_data = np.array([op1, op2, op3] + sensor_inputs).reshape(1, -1)
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Remaining Useful Life (RUL): {prediction[0]:.2f} cycles")
