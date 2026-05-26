import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
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

/* Cards */
.section-card {
    background-color: white;
    color: black;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.15);
    margin-bottom: 25px;
}

/* Force text visibility */
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

/* Upload text */
label {
    color: white !important;
}

/* Prediction box */
.prediction-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    margin-bottom: 15px;
}

/* Severity Colors */
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

/* Footer */
.footer {
    text-align: center;
    color: lightgray;
    margin-top: 30px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL ARCHITECTURE
# ---------------------------------------------------

base_model = MobileNetV2(
    weights=None,
    include_top=False,
    input_shape=(224, 224, 3)
)

x = GlobalAveragePooling2D()(base_model.output)

output = Dense(3, activation='softmax')(x)

model = Model(
    inputs=base_model.input,
    outputs=output
)

# ---------------------------------------------------
# LOAD WEIGHTS
# ---------------------------------------------------

model.load_weights("road_damage.weights.h5")

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

    img_array = img_array / 255.0

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
# HEADER SECTION
# ---------------------------------------------------

st.markdown("""
<div style="
    background: linear-gradient(90deg, #1f4e79, #3498db);
    padding: 45px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 30px;
">

<h1 style="
    color: white;
    font-size: 52px;
    margin-bottom: 10px;
">
🛣️ AI-Based Road Damage Detection System
</h1>

<p style="
    color: white;
    font-size: 24px;
">
Smart City Infrastructure Monitoring using CNN
</p>

</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# ABOUT PROJECT SECTION
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
# IMAGE UPLOAD SECTION
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
# PREDICTION PIPELINE
# ---------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    # -------------------------------------------
    # IMAGE PREVIEW
    # -------------------------------------------

    with col1:

        st.markdown("""
        <div class="section-card">
        <h3>🖼 Uploaded Image Preview</h3>
        </div>
        """, unsafe_allow_html=True)

        st.image(
            image,
            use_container_width=True
        )

    # -------------------------------------------
    # MODEL PREDICTION
    # -------------------------------------------

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image, verbose=0)

    pred_index = np.argmax(prediction)

    pred_class = CLASS_NAMES[pred_index]

    confidence = float(np.max(prediction))

    severity = get_severity(pred_class, confidence)

    recommendation = get_recommendation(pred_class)

    # -------------------------------------------
    # PREDICTION RESULTS
    # -------------------------------------------

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

        st.metric(
            label="Confidence Score",
            value=f"{confidence * 100:.2f}%"
        )

        st.metric(
            label="Severity Level",
            value=severity
        )

        st.markdown("### ⚠️ Recommendations")

        st.info(recommendation)

    # -------------------------------------------
    # VISUALIZATION SECTION
    # -------------------------------------------

    st.markdown("""
    <div class="section-card">
    <h2>📊 Class Confidence Graph</h2>
    </div>
    """, unsafe_allow_html=True)

    probabilities = prediction[0] * 100

    fig, ax = plt.subplots(figsize=(8, 4))

    bars = ax.bar(CLASS_NAMES, probabilities)

    ax.set_ylim([0, 100])

    ax.set_ylabel("Confidence (%)")

    ax.set_title("Prediction Confidence Scores")

    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.1f}%",
            ha='center'
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