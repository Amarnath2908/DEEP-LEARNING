import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.express as px
import time
import os

# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Deep Learning Fraud Detection System")
st.markdown("Upload transaction data and predict fraudulent transactions.")

# --------------------------------------------------
# Custom Layer Definition
# --------------------------------------------------

class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, max_len=100, d_model=64, **kwargs):
        # ✅ Pop any keys Keras 3.x injects that the old config didn't have
        kwargs.pop("trainable", None)
        kwargs.pop("dtype", None)
        super().__init__(**kwargs)

        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]

        angle_rates = 1 / np.power(
            10000,
            (2 * (i // 2)) / np.float32(d_model)
        )

        angles = pos * angle_rates
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])

        self.pos_encoding = tf.constant(
            angles[np.newaxis, ...],
            dtype=tf.float32
        )

        self.max_len = max_len
        self.d_model = d_model

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_len": self.max_len,
            "d_model": self.d_model
        })
        return config

    @classmethod
    def from_config(cls, config):
        # ✅ Strip unknown keys injected by newer Keras before calling __init__
        config = {
            k: v for k, v in config.items()
            if k in ("max_len", "d_model", "name")
        }
        return cls(**config)


# --------------------------------------------------
# Load Model & Scaler
# --------------------------------------------------

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "fraud_detection_model.h5")

    return tf.keras.models.load_model(
        model_path,
        custom_objects={"PositionalEncoding": PositionalEncoding},
        compile=False
    )

@st.cache_resource
def load_scaler():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    scaler_path = os.path.join(base_dir, "scaler.pkl")
    return joblib.load(scaler_path)

model = load_model()
scaler = load_scaler()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("Settings")

SEQUENCE_LENGTH = st.sidebar.slider(
    "Sequence Length",
    min_value=3,
    max_value=20,
    value=5
)

THRESHOLD = st.sidebar.slider(
    "Fraud Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.50
)

# --------------------------------------------------
# Upload CSV
# --------------------------------------------------

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# --------------------------------------------------
# Main Logic
# --------------------------------------------------

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    st.write("Shape:", df.shape)

    # Remove target column if present
    feature_df = df.copy()
    if "Class" in feature_df.columns:
        feature_df = feature_df.drop("Class", axis=1)

    # Scale Features
    try:
        scaled_features = scaler.transform(feature_df)
    except Exception as e:
        st.error(f"Feature mismatch detected.\n\n{e}")
        st.stop()

    # Create Sequences
    X = []
    for i in range(len(scaled_features) - SEQUENCE_LENGTH):
        X.append(scaled_features[i:i + SEQUENCE_LENGTH])
    X = np.array(X)

    st.success(f"{len(X)} sequences generated")

    # Predict
    predictions = model.predict(X, verbose=0)
    fraud_probs = predictions.flatten()

    result_df = pd.DataFrame({
        "Sequence_ID": np.arange(len(fraud_probs)),
        "Fraud_Probability": fraud_probs
    })

    result_df["Risk"] = np.where(
        result_df["Fraud_Probability"] >= THRESHOLD,
        "High Risk",
        "Low Risk"
    )

    # Metrics
    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sequences", len(result_df))
    with col2:
        st.metric("High Risk", len(result_df[result_df["Risk"] == "High Risk"]))
    with col3:
        st.metric("Average Fraud Probability", round(fraud_probs.mean(), 4))

    # Predictions Table
    st.subheader("Fraud Prediction Results")
    st.dataframe(result_df)

    # High Risk Transactions
    st.subheader("🚨 High Risk Transactions")
    high_risk = result_df[result_df["Risk"] == "High Risk"]
    if len(high_risk) > 0:
        st.dataframe(high_risk)
    else:
        st.success("No high-risk transactions detected.")

    # Charts
    st.subheader("Fraud Probability Trend")
    fig = px.line(result_df, x="Sequence_ID", y="Fraud_Probability",
                  title="Fraud Probability Across Sequences")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Fraud Probability Distribution")
    fig2 = px.histogram(result_df, x="Fraud_Probability", nbins=30,
                        title="Probability Distribution")
    st.plotly_chart(fig2, use_container_width=True)

    # Top 10
    st.subheader("Top 10 Highest Risk Transactions")
    top10 = result_df.sort_values(by="Fraud_Probability", ascending=False).head(10)
    st.dataframe(top10)

    # Attention Investigation
    st.subheader("Attention Investigation")
    if len(fraud_probs) > 0:
        most_important_txn = np.argmax(fraud_probs)
        st.success(f"Most Influential Transaction Sequence: {most_important_txn}")

    # Download
    st.subheader("Download Predictions")
    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Results CSV",
        data=csv,
        file_name="fraud_predictions.csv",
        mime="text/csv"
    )

    # Real-Time Simulation
    st.subheader("⚡ Real-Time Fraud Detection Simulation")
    if st.button("Start Simulation"):
        progress = st.progress(0)
        status = st.empty()
        for i in range(min(20, len(X))):
            prob = model.predict(
                np.expand_dims(X[i], axis=0), verbose=0
            )[0][0]
            status.write(f"Sequence {i} → Fraud Probability: {prob:.4f}")
            progress.progress((i + 1) / 20)
            time.sleep(0.5)
        st.success("Simulation Completed")

else:
    st.info("Upload a CSV file to begin fraud detection.")  