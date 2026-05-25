import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Titanic AI Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# ---------------------------------------------------
# LOAD MODEL & SCALER
# ---------------------------------------------------

model = tf.keras.models.load_model("titanic_ann_model1.h5")
scaler = joblib.load("scaler.pkl")

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}

.main-title {
    font-size: 50px;
    font-weight: bold;
    text-align: center;
    color: white;
}

.subtitle {
    text-align: center;
    font-size: 22px;
    color: #dddddd;
}

.card {
    background-color: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 20px;
    margin-top: 20px;
}

.metric-card {
    background-color: rgba(255,255,255,0.1);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown(
    '<p class="main-title">🚢 Titanic Survival Prediction System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Deep Learning Based Passenger Survival Prediction</p>',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("📘 About")

st.sidebar.info("""
This AI system predicts whether a passenger would survive during the Titanic disaster using:

- Artificial Neural Network (ANN)
- TensorFlow/Keras
- Deep Learning
- Streamlit Deployment
""")

# ---------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------

st.markdown('<div class="card">', unsafe_allow_html=True)

st.header("🎯 Passenger Input Form")

c1, c2, c3 = st.columns(3)

with c1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])

    sex = st.selectbox("Gender", ["male", "female"])

with c2:
    age = st.slider("Age", 1, 80, 24)

    sibsp = st.slider("Siblings/Spouse", 0, 8, 0)

with c3:
    parch = st.slider("Parents/Children", 0, 6, 0)

    fare = st.number_input("Fare", 0.0, 600.0, 100.0)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------

def preprocess():
    gender = 0 if sex == "male" else 1

    data = np.array([[
        pclass,
        gender,
        age,
        sibsp,
        parch,
        fare
    ]])

    data_scaled = scaler.transform(data)

    return data_scaled

# ---------------------------------------------------
# PREDICT BUTTON
# ---------------------------------------------------

st.markdown("")

if st.button("🚀 Predict Survival"):

    input_data = preprocess()

    prediction = model.predict(input_data)

    probability = prediction[0][0]

    survived_prob = probability
    not_survived_prob = 1 - probability

    # Prediction logic
    if probability > 0.5:
        result = "✅ SURVIVED"
    else:
        result = "❌ NOT SURVIVED"

    confidence = max(probability, 1 - probability) * 100

    # ---------------------------------------------------
    # OUTPUT SECTION
    # ---------------------------------------------------

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.header("📊 Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Prediction", result)

    with col2:
        st.metric(
            "Survival Probability",
            f"{survived_prob:.2f}"
        )

    with col3:
        st.metric(
            "Confidence Score",
            f"{confidence:.2f}%"
        )

    # ---------------------------------------------------
    # CONFIDENCE BAR
    # ---------------------------------------------------

    st.subheader("Confidence Meter")

    st.progress(float(confidence / 100))

    # ---------------------------------------------------
    # PIE CHART
    # ---------------------------------------------------

    st.subheader("Probability Visualization")

    fig, ax = plt.subplots(figsize=(5,5))

    labels = ['Survival', 'Non-Survival']
    sizes = [survived_prob, not_survived_prob]

    ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%'
    )

    st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.markdown("""
<center>
<h4>Developed using TensorFlow + Streamlit + Deep Learning</h4>
</center>
""", unsafe_allow_html=True)