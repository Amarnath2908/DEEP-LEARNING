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
        kwargs.pop("trainable", None)
        kwargs.pop("dtype", None)
        super().__init__(**kwargs)

        pos = np.arange(max_len)[:, np.newaxis]
        i   = np.arange(d_model)[np.newaxis, :]

        angle_rates = 1 / np.power(
            10000, (2 * (i // 2)) / np.float32(d_model)
        )
        angles = pos * angle_rates
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])

        self.pos_encoding = tf.constant(
            angles[np.newaxis, ...], dtype=tf.float32
        )
        self.max_len = max_len
        self.d_model  = d_model

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len, "d_model": self.d_model})
        return config

    @classmethod
    def from_config(cls, config):
        config = {k: v for k, v in config.items()
                  if k in ("max_len", "d_model", "name")}
        return cls(**config)


# --------------------------------------------------
# Helper – resolve path relative to this script
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def repo_path(filename: str) -> str:
    """Return absolute path to a file sitting next to app.py."""
    return os.path.join(BASE_DIR, filename)


# --------------------------------------------------
# File-existence check (fail fast with a clear message)
# --------------------------------------------------

MODEL_PATH  = repo_path("fraud_detection_model.keras")
SCALER_PATH = repo_path("scaler.pkl")

missing = [f for f in [MODEL_PATH, SCALER_PATH] if not os.path.exists(f)]
if missing:
    st.error(
        "**Required file(s) not found in the deployment directory:**\n\n"
        + "\n".join(f"- `{os.path.basename(f)}`" for f in missing)
        + "\n\nMake sure `fraud_detection_model.h5` **and** `scaler.pkl` are "
          "committed to the same GitHub folder as `app.py` "
          f"(`{BASE_DIR}`) and re-deploy."
    )
    st.stop()


# --------------------------------------------------
# Load Model & Scaler
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"PositionalEncoding": PositionalEncoding},
        compile=False
    )

@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)

model  = load_model()
scaler = load_scaler()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("Settings")

SEQUENCE_LENGTH = st.sidebar.slider("Sequence Length", min_value=3,  max_value=20, value=5)
THRESHOLD       = st.sidebar.slider("Fraud Threshold",  min_value=0.0, max_value=1.0, value=0.50)

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
    feature_df = df.drop("Class", axis=1) if "Class" in df.columns else df.copy()

    # Scale Features
    try:
        scaled_features = scaler.transform(feature_df)
    except Exception as e:
        st.error(f"Feature mismatch detected.\n\n{e}")
        st.stop()

    # Create Sequences
    X = np.array([
        scaled_features[i:i + SEQUENCE_LENGTH]
        for i in range(len(scaled_features) - SEQUENCE_LENGTH)
    ])

    st.success(f"{len(X)} sequences generated")

    # Predict
    fraud_probs = model.predict(X, verbose=0).flatten()

    result_df = pd.DataFrame({
        "Sequence_ID":       np.arange(len(fraud_probs)),
        "Fraud_Probability": fraud_probs,
    })
    result_df["Risk"] = np.where(
        result_df["Fraud_Probability"] >= THRESHOLD, "High Risk", "Low Risk"
    )

    # ── Metrics ────────────────────────────────────
    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sequences",          len(result_df))
    col2.metric("High Risk",                (result_df["Risk"] == "High Risk").sum())
    col3.metric("Avg Fraud Probability",    round(fraud_probs.mean(), 4))

    # ── Tables ─────────────────────────────────────
    st.subheader("Fraud Prediction Results")
    st.dataframe(result_df)

    st.subheader("🚨 High Risk Transactions")
    high_risk = result_df[result_df["Risk"] == "High Risk"]
    st.dataframe(high_risk) if len(high_risk) else st.success("No high-risk transactions detected.")

    # ── Charts ─────────────────────────────────────
    st.subheader("Fraud Probability Trend")
    st.plotly_chart(
        px.line(result_df, x="Sequence_ID", y="Fraud_Probability",
                title="Fraud Probability Across Sequences"),
        use_container_width=True
    )

    st.subheader("Fraud Probability Distribution")
    st.plotly_chart(
        px.histogram(result_df, x="Fraud_Probability", nbins=30,
                     title="Probability Distribution"),
        use_container_width=True
    )

    st.subheader("Top 10 Highest Risk Transactions")
    st.dataframe(result_df.sort_values("Fraud_Probability", ascending=False).head(10))

    # ── Attention Investigation ─────────────────────
    st.subheader("Attention Investigation")
    st.success(f"Most Influential Transaction Sequence: {np.argmax(fraud_probs)}")

    # ── Download ───────────────────────────────────
    st.subheader("Download Predictions")
    st.download_button(
        label="Download Results CSV",
        data=result_df.to_csv(index=False).encode("utf-8"),
        file_name="fraud_predictions.csv",
        mime="text/csv"
    )

    # ── Real-Time Simulation ────────────────────────
    st.subheader("⚡ Real-Time Fraud Detection Simulation")
    if st.button("Start Simulation"):
        progress = st.progress(0)
        status   = st.empty()
        n        = min(20, len(X))
        for i in range(n):
            prob = model.predict(np.expand_dims(X[i], axis=0), verbose=0)[0][0]
            status.write(f"Sequence {i} → Fraud Probability: {prob:.4f}")
            progress.progress((i + 1) / n)
            time.sleep(0.5)
        st.success("Simulation Completed")

else:
    st.info("Upload a CSV file to begin fraud detection.")