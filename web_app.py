import streamlit as st
import pandas as pd
import numpy as np
import time
from src.anomaly_detection import AnomalyDetector
from src.stream_simulation import simulate_data_stream
import matplotlib.pyplot as plt

# Set up page layout
st.set_page_config(
    page_title="Real-Time Anomaly Detection",
    layout="wide",
)

st.title("Real-Time Data Stream Anomaly Detection")
st.markdown("""
This web app simulates a real-time data stream and detects anomalies using multiple algorithms.
You can adjust the parameters for data simulation and anomaly detection, and view both the real-time data and detected anomalies.
""")

st.sidebar.header("Settings")
length = st.sidebar.slider("Stream Length", min_value=100, max_value=10000, value=1000, step=100)
noise = st.sidebar.slider("Noise Level", min_value=2, max_value=10, value=3, step=1)
algorithm = st.sidebar.selectbox("Anomaly Detection Algorithm", options=["Z-Score", "Isolation Forest"])
contamination = st.sidebar.slider("Contamination Rate (for Isolation Forest)", min_value=0.01, max_value=0.1, value=0.01, step=0.01)

# Data placeholders
st.subheader("Real-Time Data Stream (DataFrame)")
data_placeholder = st.empty()  
plot_placeholder = st.empty()  

# Initialize detector and DataFrame
detector = AnomalyDetector(contamination=contamination)
data_list = []
anomalies_list = []

# Simulate and display data stream in real-time
stream = simulate_data_stream(length=length, noise=noise)

# Fit the model initially with an initial dataset of 250 points
initial_data = simulate_data_stream(length=250, noise=2)
detector.fit([*initial_data])


for data_point in stream:
    label = None
    # Detect anomalies based on the selected algorithm
    if algorithm == "Z-Score":
        is_anomaly = detector.detect_z_score(data_point)
        label = "Anomalies (Z-Score)"
    elif algorithm == "Isolation Forest":
        is_anomaly = detector.detect_isolation_forest(data_point)
        label = "Anomalies (Isolation Forest)"
    
    
    if len(data_list) == 300:
        data_list.pop(0)  # Remove the oldest value
    data_list.append({"Value": data_point, "Anomaly": is_anomaly})
    
    # Convert to DataFrame for display
    df = pd.DataFrame(data_list)

    # Update DataFrame in UI
    data_placeholder.dataframe(df)

    # Update plot in UI
    plt.clf()
    plt.figure(figsize=(8, 4))
    plt.plot(df["Value"], label="Data Stream")
    anomaly_points = df[df["Anomaly"] == True]
    plt.plot(anomaly_points.index, anomaly_points["Value"], "ro", label=label)
    plt.legend()
    plot_placeholder.pyplot(plt)
    plt.close()