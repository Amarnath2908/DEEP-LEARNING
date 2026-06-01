# app.py


import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import string
import matplotlib.pyplot as plt

# --------------------------------------------------
# Load Models
# --------------------------------------------------

simple_rnn_model = tf.keras.models.load_model(
    "simple_rnn.keras"
)

lstm_model = tf.keras.models.load_model(
    "lstm_model.keras"
)

gru_model = tf.keras.models.load_model(
    "gru_model.keras"
)

# --------------------------------------------------
# Load Tokenizer
# --------------------------------------------------

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    layout="wide"
)

st.title("🎬 Movie Review Sentiment Analysis System")

st.subheader(
    "Deep Learning Based Sentiment Classification"
)

st.divider()

# --------------------------------------------------
# Model Selection
# --------------------------------------------------

selected_model = st.selectbox(
    "Select Model",
    [
        "SimpleRNN",
        "LSTM",
        "GRU"
    ]
)

# --------------------------------------------------
# Input Area
# --------------------------------------------------

review = st.text_area(
    "Enter your movie review here...",
    height=200
)

# --------------------------------------------------
# Preprocessing Function
# --------------------------------------------------

stop_words = set(stopwords.words("english"))

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

    seq = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        seq,
        maxlen=200,
        padding='post',
        truncating='post'
    )

    return padded

# --------------------------------------------------
# Prediction Function
# --------------------------------------------------

def predict(model, review):

    processed = preprocess(review)

    prob = model.predict(
        processed,
        verbose=0
    )[0][0]

    sentiment = (
        "Positive"
        if prob >= 0.5
        else "Negative"
    )

    confidence = (
        prob
        if prob >= 0.5
        else 1 - prob
    )

    return sentiment, confidence, prob

# --------------------------------------------------
# Predict Button
# --------------------------------------------------

if st.button("Analyze Review"):

    if review.strip() == "":

        st.warning(
            "Please enter a review."
        )

    else:

        # ------------------------------
        # Selected Model Prediction
        # ------------------------------

        if selected_model == "SimpleRNN":
            model = simple_rnn_model

        elif selected_model == "LSTM":
            model = lstm_model

        else:
            model = gru_model

        sentiment, confidence, prob = predict(
            model,
            review
        )

        st.success(
            f"Sentiment: {sentiment}"
        )

        st.info(
            f"Confidence: {confidence*100:.2f}%"
        )

        # ------------------------------
        # Probabilities
        # ------------------------------

        positive_prob = prob * 100
        negative_prob = (1 - prob) * 100

        st.subheader("Probability Analysis")

        probability_df = pd.DataFrame(
            {
                "Class":
                [
                    "Positive",
                    "Negative"
                ],
                "Probability":
                [
                    positive_prob,
                    negative_prob
                ]
            }
        )

        st.bar_chart(
            probability_df.set_index("Class")
        )

        # ------------------------------
        # Confidence Chart
        # ------------------------------

        st.subheader("Confidence Chart")

        fig, ax = plt.subplots()

        ax.bar(
            ["Confidence"],
            [confidence * 100]
        )

        ax.set_ylim(0,100)

        st.pyplot(fig)

        # ------------------------------
        # Compare All Models
        # ------------------------------

        st.subheader(
            "Comparison Across Models"
        )

        models = {
            "SimpleRNN":
            simple_rnn_model,

            "LSTM":
            lstm_model,

            "GRU":
            gru_model
        }

        results = []

        for model_name, model_obj in models.items():

            sentiment, conf, prob = predict(
                model_obj,
                review
            )

            results.append(
                [
                    model_name,
                    sentiment,
                    round(conf*100,2)
                ]
            )

        comparison_df = pd.DataFrame(
            results,
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

