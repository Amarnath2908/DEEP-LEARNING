# =========================================================
# STREAMLIT APP
# AI-BASED MENTAL HEALTH SENTIMENT MONITORING SYSTEM
# =========================================================

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================================================
# DOWNLOAD NLTK FILES
# =========================================================

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# =========================================================
# LOAD TRAINED MODEL
# =========================================================

model = load_model(
    "mental_health_rnn_model.h5"
)

# =========================================================
# LOAD TOKENIZER
# =========================================================

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# =========================================================
# LOAD LABEL ENCODER
# =========================================================

with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# =========================================================
# PARAMETERS
# =========================================================

max_length = 50

# =========================================================
# TEXT PREPROCESSING
# =========================================================

stop_words = set(stopwords.words('english'))

important_words = ['not', 'no', 'never']

def preprocess_text(text):

    # Lowercase
    text = text.lower()

    # Remove punctuation/numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization
    words = word_tokenize(text)

    # Stopword removal
    words = [
        word for word in words
        if word not in stop_words
        or word in important_words
    ]

    # Join words
    cleaned_text = " ".join(words)

    return cleaned_text

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Mental Health Sentiment Monitoring",
    page_icon="🧠",
    layout="wide"
)

# =========================================================
# SECTION 1 — HEADER
# =========================================================

st.title(
    "🧠 AI-Based Mental Health Sentiment Monitoring System"
)

st.subheader(
    "Emotion Detection using Simple Recurrent Neural Networks"
)

st.markdown("---")

# =========================================================
# SECTION 2 — ABOUT PROJECT
# =========================================================

st.header("📘 About the Project")

st.write("""

This AI-powered application analyzes user text messages
to detect emotional sentiment patterns using Deep Learning
and Natural Language Processing (NLP).

### Importance of Emotional AI
Emotional AI helps identify:
- stress
- anxiety
- depression
- emotional distress
- suicidal tendencies

It supports early intervention and emotional wellness monitoring.

### NLP Applications
Natural Language Processing enables machines to:
- understand human emotions
- analyze text sentiment
- process sequential language data
- generate intelligent predictions

### Role of RNN/LSTM in Sequence Learning
RNNs and LSTMs process words sequentially while remembering
previous context using hidden states and memory cells.

This helps the model understand emotional meaning from sentences.

""")

st.markdown("---")

# =========================================================
# SECTION 3 — USER INPUT AREA
# =========================================================

st.header("📝 Enter Your Thoughts")

st.write("### Sample Sentences")

st.info("""
• I feel mentally exhausted and hopeless.
• Nobody understands my emotions.
• I feel happy and motivated today.
• I am anxious about my future.
""")

user_input = st.text_area(

    "Enter your thoughts or feelings here...",

    height=200,

    placeholder="""
Example:
I feel stressed and emotionally tired lately...
"""
)

st.markdown("---")

# =========================================================
# SECTION 4 — PREDICTION BUTTON
# =========================================================

analyze_button = st.button(
    "🔍 Analyze Emotion"
)

# =========================================================
# SECTION 5 — PREDICTION OUTPUT
# =========================================================

if analyze_button:

    if user_input.strip() == "":

        st.warning("Please enter some text.")

    else:

        # =============================================
        # PREPROCESS INPUT
        # =============================================

        cleaned = preprocess_text(user_input)

        # =============================================
        # TOKENIZATION
        # =============================================

        sequence = tokenizer.texts_to_sequences(
            [cleaned]
        )

        # =============================================
        # PADDING
        # =============================================

        padded = pad_sequences(
            sequence,
            maxlen=max_length,
            padding='post',
            truncating='post'
        )

        # =============================================
        # PREDICTION
        # =============================================

        prediction = model.predict(padded)

        predicted_class = np.argmax(prediction)

        confidence = np.max(prediction) * 100

        emotion = encoder.inverse_transform(
            [predicted_class]
        )[0]

        # =============================================
        # DISPLAY RESULTS
        # =============================================

        st.header("📊 Prediction Output")

        st.success(
            f"Emotion Detected: {emotion}"
        )

        st.info(
            f"Confidence Score: {confidence:.2f}%"
        )

        # =============================================
        # EMOTIONAL STATUS
        # =============================================

        if emotion.lower() in [
            'depression',
            'suicidal',
            'stress',
            'anxiety'
        ]:

            st.error(
                "Emotional Status: Emotional distress detected."
            )

        elif emotion.lower() == 'normal':

            st.success(
                "Emotional Status: Emotionally stable."
            )

        else:

            st.warning(
                "Emotional Status: Emotional pattern identified."
            )

        st.markdown("---")

        # =================================================
        # SECTION 6 — VISUALIZATION AREA
        # =================================================

        st.header("📈 Sentiment Confidence Graph")

        labels = encoder.classes_

        probabilities = prediction[0] * 100

        chart_data = pd.DataFrame({
            "Emotion": labels,
            "Confidence": probabilities
        })

        st.bar_chart(
            chart_data.set_index("Emotion")
        )

        # =================================================
        # SECTION 7 — EMOTIONAL GUIDANCE
        # =================================================

        st.header("💡 Emotional Wellness Guidance")

        if emotion.lower() == "anxiety":

            st.warning("""
Take deep breaths and try relaxation exercises.
Talking to someone you trust may help reduce anxiety.
""")

        elif emotion.lower() == "depression":

            st.warning("""
Remember that difficult moments are temporary.
Consider reaching out to supportive friends or professionals.
""")

        elif emotion.lower() == "stress":

            st.warning("""
Take a short break and focus on self-care.
Proper sleep and relaxation can help reduce stress.
""")

        elif emotion.lower() == "suicidal":

            st.error("""
You are not alone.
Please talk to a trusted person or mental health professional immediately.
Seeking help is important.
""")

        elif emotion.lower() == "normal":

            st.success("""
Great to hear positive emotional stability.
Continue maintaining healthy habits and positivity.
""")

        else:

            st.info("""
Stay positive and practice healthy emotional habits.
Mindfulness and social support are beneficial.
""")

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Developed using TensorFlow, NLP, Bidirectional LSTM, and Streamlit"
)