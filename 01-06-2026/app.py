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
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    layout="wide"
)

# =====================================================
# NLTK DOWNLOAD
# =====================================================

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# =====================================================
# SIDEBAR DEBUG INFO
# =====================================================

st.sidebar.title("System Information")

st.sidebar.write(
    f"TensorFlow Version: {tf.__version__}"
)

try:
    st.sidebar.write("Files Found:")
    st.sidebar.write(os.listdir("."))
except Exception as e:
    st.sidebar.error(str(e))

# =====================================================
# LOAD MODELS
# =====================================================

simple_rnn_model = None
lstm_model = None
gru_model = None

try:
    simple_rnn_model = tf.keras.models.load_model(
        "simple_rnn.keras",
        compile=False
    )

    st.sidebar.success(
        "✅ SimpleRNN Loaded"
    )

except Exception as e:

    st.sidebar.error(
        f"SimpleRNN Error:\n{str(e)}"
    )

try:
    lstm_model = tf.keras.models.load_model(
        "lstm_model.keras",
        compile=False
    )

    st.sidebar.success(
        "✅ LSTM Loaded"
    )

except Exception as e:

    st.sidebar.error(
        f"LSTM Error:\n{str(e)}"
    )

try:
    gru_model = tf.keras.models.load_model(
        "gru_model.keras",
        compile=False
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

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    st.sidebar.success(
        "✅ Tokenizer Loaded"
    )

except Exception as e:

    st.error(
        f"Tokenizer Loading Error:\n{str(e)}"
    )

    st.stop()

# =====================================================
# MAIN UI
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
# REVIEW INPUT
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
            '',
            '',
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

def predict_sentiment(model, review_text):

    processed_review = preprocess(
        review_text
    )

    probability = model.predict(
        processed_review,
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
# BUTTON
# =====================================================

if st.button("Analyze Review"):

    if review.strip() == "":

        st.warning(
            "Please enter a movie review."
        )

    else:

        # ------------------------------------------
        # SELECT MODEL
        # ------------------------------------------

        if selected_model == "SimpleRNN":
            model = simple_rnn_model

        elif selected_model == "LSTM":
            model = lstm_model

        else:
            model = gru_model

        if model is None:

            st.error(
                f"{selected_model} failed to load."
            )

            st.stop()

        sentiment, confidence, prob = predict_sentiment(
            model,
            review
        )

        # ------------------------------------------
        # OUTPUT
        # ------------------------------------------

        st.success(
            f"Sentiment: {sentiment}"
        )

        st.info(
            f"Confidence: {confidence * 100:.2f}%"
        )

        # ------------------------------------------
        # PROBABILITIES
        # ------------------------------------------

        positive_prob = prob * 100
        negative_prob = (1 - prob) * 100

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
                    positive_prob,
                    negative_prob
                ]
            }
        )

        st.bar_chart(
            probability_df.set_index(
                "Class"
            )
        )

        # ------------------------------------------
        # CONFIDENCE CHART
        # ------------------------------------------

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

        # ------------------------------------------
        # COMPARE ALL MODELS
        # ------------------------------------------

        st.subheader(
            "Compare All Models"
        )

        comparison_results = []

        all_models = {
            "SimpleRNN": simple_rnn_model,
            "LSTM": lstm_model,
            "GRU": gru_model
        }

        for model_name, model_obj in all_models.items():

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