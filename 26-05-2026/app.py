import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # ✅ FIX
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🛣️",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #0b1120;
}

.section-card {
    background-color: white;
    color: black;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.15);
    margin-bottom: 25px;
}

.section-card h1,
.section-card h2,
.section-card h3,
.section-card h4,
.section-card h5,
.section-card h6,
.section-card p,
.section-card li {
    color: black !important;
}

label {
    color: white !important;
}

.prediction-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    margin-bottom: 15px;
}

.high {
    background-color: #ffcccc;
    color: #b30000;
}

.medium {
    background-color: #fff3cd;
    color: #856404;
}

.low {
    background-color: #d4edda;
    color: #155724;
}

.footer {
    text-align: center;
    color: lightgray;
    margin-top: 30px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

# ✅ FIX 1: Use path relative to app.py so it works on Streamlit Cloud

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "road_damage_model.keras")

@st.cache_resource
def load_model():
    # ✅ Loads architecture + weights together — no mismatch possible
    return tf.keras.models.load_model(model_path)

model = load_model()

CLASS_NAMES = ['crack', 'manhole', 'pothole']   

# ---------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------

def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image)

    # Remove alpha channel if present
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = img_array.astype(np.float32)

    # ✅ FIX 3: Use MobileNetV2's preprocess_input instead of /255
    # preprocess_input scales pixels to [-1, 1] as MobileNetV2 expects
    img_array = preprocess_input(img_array)

    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def get_severity(pred_class, confidence):
    if pred_class == "pothole":
        if confidence > 0.85:
            return "High"
        elif confidence > 0.60:
            return "Medium"
        else:
            return "Low"
    elif pred_class == "crack":
        if confidence > 0.80:
            return "Medium"
        else:
            return "Low"
    return "Low"


def get_recommendation(pred_class):
    if pred_class == "pothole":
        return """
🚨 Immediate maintenance recommended.

⚠️ High-risk road condition detected.

🚧 Repair priority should be HIGH.
"""
    elif pred_class == "crack":
        return """
🛠 Schedule road inspection soon.

📌 Surface deterioration detected.

🚧 Preventive maintenance recommended.
"""
    elif pred_class == "manhole":
        return """
✅ Manhole detected.

🔍 Ensure proper alignment and cover safety.
"""
    return "No recommendation available."


# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown("""
<div style="
    background: linear-gradient(90deg, #1f4e79, #3498db);
    padding: 45px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 30px;
">
<h1 style="color: white; font-size: 52px; margin-bottom: 10px;">
🛣️ AI-Based Road Damage Detection System
</h1>
<p style="color: white; font-size: 24px;">
Smart City Infrastructure Monitoring using CNN
</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# ABOUT
# ---------------------------------------------------

st.markdown("""
<div class="section-card">
<h2>📌 About the Project</h2>
<p>
Road damage monitoring is essential for maintaining safe transportation systems.
Damaged roads can cause accidents, traffic congestion, and vehicle damage.
</p>
<p>
This project uses <b>Convolutional Neural Networks (CNN)</b> with MobileNetV2
to automatically detect road damages such as potholes, cracks, and manholes.
</p>
<h3>🚀 Industry Applications</h3>
<ul>
<li>Smart City Monitoring</li>
<li>Automated Road Inspection</li>
<li>Municipal Infrastructure Management</li>
<li>AI-Based Maintenance Planning</li>
<li>Road Safety Systems</li>
</ul>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# UPLOAD
# ---------------------------------------------------

st.markdown("""
<div class="section-card">
<h2>📤 Upload Road Image</h2>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a road image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        <div class="section-card">
        <h3>🖼 Uploaded Image Preview</h3>
        </div>
        """, unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image, verbose=0)
    pred_index = np.argmax(prediction)
    pred_class = CLASS_NAMES[pred_index]
    confidence = float(np.max(prediction))
    severity = get_severity(pred_class, confidence)
    recommendation = get_recommendation(pred_class)

    with col2:
        st.markdown("""
        <div class="section-card">
        <h3>🔍 Prediction Result</h3>
        </div>
        """, unsafe_allow_html=True)

        severity_class = severity.lower()

        st.markdown(f"""
        <div class="prediction-box {severity_class}">
            {pred_class.upper()} DETECTED
        </div>
        """, unsafe_allow_html=True)

        st.metric(label="Confidence Score", value=f"{confidence * 100:.2f}%")
        st.metric(label="Severity Level", value=severity)

        st.markdown("### ⚠️ Recommendations")
        st.info(recommendation)

    # Confidence bar chart
    st.markdown("""
    <div class="section-card">
    <h2>📊 Class Confidence Graph</h2>
    </div>
    """, unsafe_allow_html=True)

    probabilities = prediction[0] * 100
    colors = ['#e74c3c' if cn == pred_class else '#3498db' for cn in CLASS_NAMES]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(CLASS_NAMES, probabilities, color=colors)
    ax.set_ylim([0, 110])
    ax.set_ylabel("Confidence (%)")
    ax.set_title("Prediction Confidence Scores")

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.1f}%",
            ha='center',
            fontweight='bold'
        )

    st.pyplot(fig)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("""
<div class="footer">
Developed using Streamlit • TensorFlow • MobileNetV2
</div>
""", unsafe_allow_html=True)