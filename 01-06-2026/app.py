import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
import os
import nltk
import string
import matplotlib.pyplot as plt

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =====================================================
# CONFIGURATION
# =====================================================

MODEL_DIR = "01-06-2026"
st.sidebar.write("Current Working Directory")
st.sidebar.write(os.getcwd())

if os.path.exists(MODEL_DIR):
    st.sidebar.success(f"{MODEL_DIR} folder found")
    st.sidebar.write(os.listdir(MODEL_DIR))
else:
    st.sidebar.error(f"{MODEL_DIR} folder not found")

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    layout="wide"
)

# =====================================================
# DOWNLOAD NLTK DATA
# =====================================================

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# =====================================================
# DEBUG INFORMATION
# =====================================================

st.sidebar.title("System Information")

st.sidebar.write(
    f"TensorFlow Version: {tf.__version__}"
)

st.sidebar.write("Root Folder Files")

try:
    st.sidebar.write(os.listdir("."))
except Exception as e:
    st.sidebar.error(str(e))

st.sidebar.write("Model Folder Files")

try:
    st.sidebar.write(os.listdir(MODEL_DIR))
except Exception as e:
    st.sidebar.error(str(e))

# =====================================================
# LOAD MODELS
# =====================================================

simple_rnn_model = None
lstm_model = None
gru_model = None

try:

    simple_rnn_path = os.path.join(
    MODEL_DIR,
    "simple_rnn.keras"
)

    if os.path.exists(simple_rnn_path):
            simple_rnn_model = tf.keras.models.load_model(
            simple_rnn_path,
            compile=False
        )
    else:
        st.sidebar.error(
        f"Missing file: {simple_rnn_path}"
    )

    st.sidebar.success(
        "✅ SimpleRNN Loaded"
    )

except Exception as e:

    st.sidebar.error(
        f"SimpleRNN Error:\n{str(e)}"
    )

try:

    lstm_path = os.path.join(
    MODEL_DIR,
    "simple_rnn.keras"
)

    if os.path.exists(lstm_path):
            simple_rnn_model = tf.keras.models.load_model(
            simple_rnn_path,
            compile=False
        )
    else:
        st.sidebar.error(
        f"Missing file: {lstm_path}"
    )

    st.sidebar.success(
        "✅ LSTM Loaded"
    )

except Exception as e:

    st.sidebar.error(
        f"LSTM Error:\n{str(e)}"
    )

try:

    gru_path = os.path.join(
    MODEL_DIR,
    "simple_rnn.keras"
)

    if os.path.exists(gru_path):
            simple_rnn_model = tf.keras.models.load_model(
            simple_rnn_path,
            compile=False
        )
    else:
        st.sidebar.error(
        f"Missing file: {gru_path}"
    )

    st.sidebar.success(
        "✅ GRU Loaded"
    )

except Exception as e:

    st.sidebar.error(
        f"GRU Error:\n{str(e)}"
    )

# =====================================================
# LOAD TOKENIZER
# =====================================================

try:

    tokenizer_path = os.path.join(
    MODEL_DIR,
    "tokenizer.pkl"
)

    if os.path.exists(tokenizer_path):

        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)

        st.sidebar.success(
        "✅ Tokenizer Loaded"
    )

    else:
        st.sidebar.error(
        f"Tokenizer not found: {tokenizer_path}"
    )
    st.stop()

    tokenizer = pickle.load(f)

    st.sidebar.success(
        "✅ Tokenizer Loaded"
    )

except Exception as e:

    st.error(
        f"Tokenizer Error:\n{str(e)}"
    )

    st.stop()

# =====================================================
# UI
# =====================================================

st.title(
    "🎬 Movie Review Sentiment Analysis System"
)

st.subheader(
    "Deep Learning Based Sentiment Classification"
)

st.markdown("---")

# =====================================================
# MODEL SELECTION
# =====================================================

selected_model = st.selectbox(
    "Select Model",
    [
        "SimpleRNN",
        "LSTM",
        "GRU"
    ]
)

# =====================================================
# INPUT AREA
# =====================================================

review = st.text_area(
    "Enter your movie review here...",
    height=200
)

# =====================================================
# STOPWORDS
# =====================================================

stop_words = set(
    stopwords.words("english")
)

# =====================================================
# PREPROCESSING
# =====================================================

def preprocess(text):

    text = text.lower()

    text = BeautifulSoup(
        text,
        "html.parser"
    ).get_text()

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    tokens = text.split()

    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    text = " ".join(tokens)

    sequence = tokenizer.texts_to_sequences(
        [text]
    )

    padded = pad_sequences(
        sequence,
        maxlen=200,
        padding="post",
        truncating="post"
    )

    return padded

# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict_sentiment(model, text):

    processed = preprocess(text)

    probability = model.predict(
        processed,
        verbose=0
    )[0][0]

    sentiment = (
        "Positive"
        if probability >= 0.5
        else "Negative"
    )

    confidence = (
        probability
        if probability >= 0.5
        else 1 - probability
    )

    return sentiment, confidence, probability

# =====================================================
# PREDICT BUTTON
# =====================================================

if st.button("Analyze Review"):

    if review.strip() == "":

        st.warning(
            "Please enter a movie review."
        )

    else:

        if selected_model == "SimpleRNN":
            selected_loaded_model = simple_rnn_model

        elif selected_model == "LSTM":
            selected_loaded_model = lstm_model

        else:
            selected_loaded_model = gru_model

        if selected_loaded_model is None:

            st.error(
                f"{selected_model} model is unavailable."
            )

            st.stop()

        sentiment, confidence, probability = predict_sentiment(
            selected_loaded_model,
            review
        )

        # ==========================================
        # OUTPUT
        # ==========================================

        st.success(
            f"Sentiment: {sentiment}"
        )

        st.info(
            f"Confidence: {confidence*100:.2f}%"
        )

        # ==========================================
        # PROBABILITIES
        # ==========================================

        positive_probability = probability * 100
        negative_probability = (
            1 - probability
        ) * 100

        st.subheader(
            "Probability Analysis"
        )

        probability_df = pd.DataFrame(
            {
                "Class": [
                    "Positive",
                    "Negative"
                ],
                "Probability": [
                    positive_probability,
                    negative_probability
                ]
            }
        )

        st.bar_chart(
            probability_df.set_index(
                "Class"
            )
        )

        # ==========================================
        # CONFIDENCE CHART
        # ==========================================

        st.subheader(
            "Confidence Chart"
        )

        fig, ax = plt.subplots()

        ax.bar(
            ["Confidence"],
            [confidence * 100]
        )

        ax.set_ylim(0, 100)

        ax.set_ylabel(
            "Confidence (%)"
        )

        st.pyplot(fig)

        # ==========================================
        # COMPARE ALL MODELS
        # ==========================================

        st.subheader(
            "Compare Predictions Across Models"
        )

        comparison_results = []

        models = {
            "SimpleRNN": simple_rnn_model,
            "LSTM": lstm_model,
            "GRU": gru_model
        }

        for model_name, model_obj in models.items():

            if model_obj is None:
                continue

            pred_sentiment, pred_confidence, _ = predict_sentiment(
                model_obj,
                review
            )

            comparison_results.append(
                [
                    model_name,
                    pred_sentiment,
                    round(
                        pred_confidence * 100,
                        2
                    )
                ]
            )

        comparison_df = pd.DataFrame(
            comparison_results,
            columns=[
                "Model",
                "Prediction",
                "Confidence (%)"
            ]
        )

        st.dataframe(
            comparison_df,
            use_container_width=True
        )