# =========================================================
# AI-BASED MENTAL HEALTH SENTIMENT MONITORING SYSTEM
# STREAMLIT DEPLOYMENT
# =========================================================

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import os
import re
import pickle
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import nltk

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# =========================================================
# DOWNLOAD NLTK FILES
# =========================================================

nltk.download('punkt')
nltk.download('stopwords')

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Mental Health Sentiment Monitoring",
    page_icon="🧠",
    layout="wide"
)

# =========================================================
# GET CURRENT DIRECTORY
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# FILE PATHS
# =========================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "mental_health_rnn_model.h5"
)

TOKENIZER_PATH = os.path.join(
    BASE_DIR,
    "tokenizer.pkl"
)

LABEL_ENCODER_PATH = os.path.join(
    BASE_DIR,
    "label_encoder.pkl"
)

# =========================================================
# CHECK FILE EXISTENCE
# =========================================================

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found!")
    st.stop()

if not os.path.exists(TOKENIZER_PATH):
    st.error("Tokenizer file not found!")
    st.stop()

if not os.path.exists(LABEL_ENCODER_PATH):
    st.error("Label encoder file not found!")
    st.stop()

# =========================================================
# LOAD MODEL
# =========================================================

model = load_model(MODEL_PATH)

# =========================================================
# LOAD TOKENIZER
# =========================================================

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# =========================================================
# LOAD LABEL ENCODER
# =========================================================

with open(LABEL_ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

# =========================================================
# PARAMETERS
# =========================================================

MAX_LENGTH = 50

stop_words = set(stopwords.words('english'))

important_words = ['not', 'no', 'never']

# =========================================================
# TEXT PREPROCESSING FUNCTION
# =========================================================

def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization
    words = word_tokenize(text)

    # Remove stopwords
    words = [
        word for word in words
        if word not in stop_words
        or word in important_words
    ]

    # Join cleaned words
    cleaned_text = " ".join(words)

    return cleaned_text

# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_emotion(text):

    # Preprocess text
    cleaned = preprocess_text(text)

    # Convert to sequence
    sequence = tokenizer.texts_to_sequences([cleaned])

    # Padding
    padded = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding='post',
        truncating='post'
    )

    # Predict
    prediction = model.predict(padded)

    # Predicted class
    predicted_class = np.argmax(prediction)

    # Confidence score
    confidence = np.max(prediction) * 100

    # Emotion label
    emotion = encoder.inverse_transform(
        [predicted_class]
    )[0]

    return emotion, confidence, prediction[0]

# =========================================================
# SECTION 1 — HEADER
# =========================================================

st.title(
    "AI-Based Mental Health Sentiment Monitoring System"
)

st.subheader(
    "Emotion Detection using Simple Recurrent Neural Networks"
)

st.markdown("---")

# =========================================================
# SECTION 2 — ABOUT PROJECT
# =========================================================

st.header("About the Project")

st.write("""
Emotional Artificial Intelligence (AI) helps machines
understand human emotions from textual data.

Natural Language Processing (NLP) is widely used in:
- sentiment analysis
- healthcare systems
- recommendation systems
- chatbots
- social media analysis

Recurrent Neural Networks (RNNs) are sequence learning
models that remember previous information in text,
making them highly effective for emotional analysis.
""")

st.markdown("---")

# =========================================================
# SECTION 3 — USER INPUT AREA
# =========================================================

st.header("User Text Input")

st.write("### Sample Sentence Suggestions")

st.info("I feel lonely and emotionally exhausted.")

st.info("Nobody understands how stressed I feel.")

st.info("I am very happy and excited today.")

st.info("I feel mentally tired and anxious.")

user_input = st.text_area(
    "Enter Text",
    placeholder="Enter your thoughts or feelings here...",
    height=200
)

st.markdown("---")

# =========================================================
# SECTION 4 — PREDICTION BUTTON
# =========================================================

analyze = st.button("Analyze Emotion")

# =========================================================
# SECTION 5 — PREDICTION OUTPUT
# =========================================================

if analyze:

    if user_input.strip() == "":

        st.warning("Please enter some text.")

    else:

        # Prediction
        emotion, confidence, probabilities = predict_emotion(
            user_input
        )

        st.header("Prediction Output")

        st.success(
            f"Emotion Detected: {emotion}"
        )

        st.info(
            f"Confidence Score: {confidence:.2f}%"
        )

        # Emotional status
        if emotion.lower() in [
            'depression',
            'stress',
            'anxiety'
        ]:

            emotional_status = (
                "User may require emotional support."
            )

        elif emotion.lower() == 'suicidal':

            emotional_status = (
                "User may require immediate emotional attention."
            )

        elif emotion.lower() == 'normal':

            emotional_status = (
                "User appears emotionally stable."
            )

        else:

            emotional_status = (
                "Emotional pattern detected."
            )

        st.warning(
            f"Emotional Status: {emotional_status}"
        )

        st.markdown("---")

        # =================================================
        # SECTION 6 — VISUALIZATION AREA
        # =================================================

        st.header("Visualization Area")

        labels = encoder.classes_

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.bar(labels, probabilities)

        ax.set_title(
            "Sentiment Confidence Graph"
        )

        ax.set_xlabel(
            "Emotion Categories"
        )

        ax.set_ylabel(
            "Probability"
        )

        plt.xticks(rotation=45)

        st.pyplot(fig)

        st.markdown("---")

        # =================================================
        # SECTION 7 — EMOTIONAL GUIDANCE AREA
        # =================================================

        st.header("Emotional Guidance Area")

        if emotion.lower() in [
            'anxiety',
            'stress',
            'depression'
        ]:

            st.error(
                "Take a short break and talk with someone you trust."
            )

            st.write("""
            ### Positive Activities
            - Practice meditation
            - Listen to calming music
            - Go for a short walk
            - Stay hydrated
            - Talk with supportive people

            ### Emotional Wellness Tips
            - Maintain healthy sleep
            - Reduce overthinking
            - Practice deep breathing
            - Take regular breaks
            """)

        elif emotion.lower() == 'suicidal':

            st.error(
                "Please seek immediate professional emotional support."
            )

            st.write("""
            ### Important Guidance
            - Contact trusted family or friends
            - Reach out to mental health professionals
            - Avoid isolation
            - Seek emergency support if needed
            """)

        elif emotion.lower() == 'normal':

            st.success(
                "Great! Maintain your positive emotional balance."
            )

            st.write("""
            ### Positive Activities
            - Continue healthy habits
            - Exercise regularly
            - Stay socially active
            - Practice gratitude

            ### Wellness Tips
            - Maintain work-life balance
            - Continue mindfulness exercises
            """)

        else:

            st.info(
                "Stay positive and continue taking care of your emotional wellness."
            )

            st.write("""
            ### Wellness Suggestions
            - Spend time with loved ones
            - Maintain daily routines
            - Take proper rest
            - Stay physically active
            """)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "AI-Based Mental Health Sentiment Monitoring System using NLP and RNN"
)

