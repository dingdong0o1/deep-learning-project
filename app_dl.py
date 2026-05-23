import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

st.set_page_config(page_title="Deep Learning Project", page_icon="🧠")

st.title("🧠 Deep Learning Project")
st.markdown("*Spam Detection + Sentiment Analysis using LSTM & BiLSTM*")

# Check if models exist
if not os.path.exists("models_dl/spam_lstm_model.h5"):
    st.error("❌ Models not found! Please run Colab first and copy models to 'models_dl' folder")
    st.stop()

@st.cache_resource
def load_models():
    spam_model = tf.keras.models.load_model("models_dl/spam_lstm_model.h5")
    with open("models_dl/spam_tokenizer.pkl", "rb") as f:
        spam_tokenizer = pickle.load(f)
    
    sentiment_model = tf.keras.models.load_model("models_dl/sentiment_bilstm_model.h5")
    with open("models_dl/sentiment_tokenizer.pkl", "rb") as f:
        sentiment_tokenizer = pickle.load(f)
    
    return spam_model, spam_tokenizer, sentiment_model, sentiment_tokenizer

spam_model, spam_tokenizer, sentiment_model, sentiment_tokenizer = load_models()

st.success("✅ Models loaded successfully!")

user_input = st.text_area("Enter your text:", height=100, placeholder="Type something here...")

col1, col2 = st.columns(2)

with col1:
    if st.button("🔍 Check Spam/Ham", type="primary"):
        if user_input:
            seq = spam_tokenizer.texts_to_sequences([user_input])
            pad = pad_sequences(seq, maxlen=100)
            pred = spam_model.predict(pad, verbose=0)[0][0]
            if pred > 0.5:
                st.error(f"🚨 **SPAM** ({pred:.2%})")
            else:
                st.success(f"✅ **HAM** ({(1-pred):.2%})")
        else:
            st.warning("⚠️ Please enter some text first!")

with col2:
    if st.button("😊 Check Sentiment", type="primary"):
        if user_input:
            seq = sentiment_tokenizer.texts_to_sequences([user_input])
            pad = pad_sequences(seq, maxlen=200)
            pred = sentiment_model.predict(pad, verbose=0)[0][0]
            if pred > 0.5:
                st.success(f"😊 **POSITIVE** ({pred:.2%})")
            else:
                st.error(f"😞 **NEGATIVE** ({(1-pred):.2%})")
        else:
            st.warning("⚠️ Please enter some text first!")

st.markdown("---")
st.caption("🔹 LSTM Neural Networks | 🔹 TensorFlow | 🔹 Streamlit")