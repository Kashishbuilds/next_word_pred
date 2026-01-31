import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Load Model and Tokenizer ---
model = load_model("lstm_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("max_len.pkl", "rb") as f:
    max_len = pickle.load(f)

# --- Prediction Function ---
def predict_next_word(text):
    text_seq = tokenizer.texts_to_sequences([text])
    text_pad = pad_sequences(text_seq, maxlen=max_len-1, padding='pre')
    predicted = np.argmax(model.predict(text_pad), axis=-1)
    
    # Convert predicted index to word
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            return word
    return ""

# --- Streamlit UI ---
st.title("Next Word Predictor 🌟")
st.write("Type a sentence, and the model will predict the next word!")

# User Input
user_input = st.text_input("Enter your text here:")

# Predict Button
if st.button("Predict Next Word"):
    if user_input.strip() != "":
        next_word = predict_next_word(user_input)
        st.success(f"Predicted Next Word: **{next_word}**")
    else:
        st.error("Please enter some text to predict!")
