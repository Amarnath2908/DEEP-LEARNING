import os
import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re
import pandas as pdc
import plotly.express as px

from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Medical Specialty Classifier",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------------
# CUSTOM CSS
# -----------------------------------
st.markdown("""
<style>

/* Main App Background */
.stApp {
    background-color: #0E1117;
    color: white;
}

/* Title */
.big-title {
    font-size: 100px;
    font-weight: 600;
    color: #4FC3F7;
    text-align: center;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #B0BEC5;
    font-size: 26px;
    font-weight: 400;
    margin-top: 6px;
}

/* Prediction Card */
.pred-box {
    background: linear-gradient(
        135deg,
        #1E293B,
        #111827
    );
    padding: 25px;
    border-radius: 20px;
    border-left: 6px solid #38BDF8;
    color: white;
    box-shadow: 0px 0px 20px rgba(56,189,248,0.2);
}

/* Prediction Heading */
.pred-box h2 {
    color: #CBD5E1;
}

/* Predicted Specialty */
.pred-box h1 {
    color: #4FC3F7;
}

/* Confidence */
.pred-box h3 {
    color: #E2E8F0;
}

/* Text Area */
textarea {
    background-color: #1E293B !important;
    color: white !important;
    border-radius: 10px !important;
    border: 1px solid #334155 !important;
}

/* Buttons */
.stButton > button {
    width: 100%;
    background: linear-gradient(
        90deg,
        #2563EB,
        #06B6D4
    );
    color: white;
    border: none;
    border-radius: 12px;
    height: 50px;
    font-size: 18px;
    font-weight: 600;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background-color: #111827;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# LOAD MODEL
# -----------------------------------

@st.cache_resource
def load_artifacts():
    # Build absolute paths relative to this script — fixes FileNotFoundError on Streamlit Cloud
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model = tf.keras.models.load_model(
        os.path.join(BASE_DIR, "medical_specialty_model.h5"),
        compile=False
    )

    with open(os.path.join(BASE_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)

    with open(os.path.join(BASE_DIR, "label_encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)

    return model, tokenizer, encoder

model, tokenizer, encoder = load_artifacts()

MAX_LEN = 300

# -----------------------------------
# CLEANING FUNCTION
# -----------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return text

# -----------------------------------
# PREDICTION FUNCTION
# -----------------------------------

def predict_specialty(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    pred = model.predict(seq, verbose=0)
    class_idx = np.argmax(pred)
    specialty = encoder.inverse_transform([class_idx])[0]
    confidence = np.max(pred)
    return specialty, confidence, pred[0]

# -----------------------------------
# HEADER
# -----------------------------------

st.markdown(
    '<p class="big-title">🩺 Medical Specialty Classifier</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Transformer-Based Clinical Text Classification</p>',
    unsafe_allow_html=True
)

st.divider()

# -----------------------------------
# SIDEBAR
# -----------------------------------

with st.sidebar:
    st.header("📋 About")
    st.write("""
    This AI model predicts the medical specialty
    from clinical reports.

    Examples:

    - Neurology
    - Cardiology
    - Orthopedics
    - Radiology
    - Gastroenterology
    """)
    st.success(f"Total Classes: {len(encoder.classes_)}")

# -----------------------------------
# INPUT AREA
# -----------------------------------

col1, col2 = st.columns([2, 1])

with col1:
    report = st.text_area(
        "Enter Clinical Report",
        height=250,
        placeholder="""
Patient suffered acute stroke involving
left cerebral hemisphere with weakness.
"""
    )

with col2:
    st.subheader("Example Reports")

    if st.button("Neurology Example"):
        st.session_state.example = """
Patient presented with acute stroke,
hemiparesis and weakness in left side.
"""

    if st.button("Cardiology Example"):
        st.session_state.example = """
Patient experiencing chest pain,
elevated troponin levels and myocardial infarction.
"""

    if st.button("Orthopedic Example"):
        st.session_state.example = """
Patient has fractured femur after
motor vehicle accident.
"""

# Load example into text box
if "example" in st.session_state:
    report = st.session_state.example

# -----------------------------------
# PREDICT BUTTON
# -----------------------------------

if st.button("🔍 Predict Specialty"):

    if len(report.strip()) == 0:
        st.warning("Please enter a medical report.")

    else:
        specialty, confidence, probs = predict_specialty(report)

        st.markdown(
            f"""
            <div class='pred-box'>
            <h2>Predicted Specialty</h2>
            <h1>{specialty}</h1>
            <h3>Confidence: {confidence*100:.2f}%</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.metric(
            label="Confidence Score",
            value=f"{confidence*100:.2f}%"
        )

        st.subheader("Prediction Probabilities")

        prob_df = pd.DataFrame({
            "Specialty": encoder.classes_,
            "Probability": probs
        })

        prob_df = prob_df.sort_values(
            "Probability",
            ascending=False
        ).head(10)

        # Fixed: plotly import and chart code now correctly inside the else block
        fig = px.bar(
            prob_df,
            x="Probability",
            y="Specialty",
            orientation="h",
            title="Top Predicted Specialties"
        )

        fig.update_layout(
            template="plotly_dark",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(prob_df, use_container_width=True)

st.divider()

st.caption("Built using TensorFlow, Transformers and Streamlit")