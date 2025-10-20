import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import nltk
from nltk.corpus import stopwords
import numpy as np

# --- Pre-load necessary items ---
# Download stopwords if not already present
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# Load the trained model using the new .keras format
# Using st.cache_resource to load the model only once
@st.cache_resource
def load_model():
    # Make sure the filename matches the one you downloaded
    return tf.keras.models.load_model('emoji_predictor_model.keras')

model = load_model()

# Load the tokenizer
@st.cache_data
def load_tokenizer():
    with open('tokenizer.pickle', 'rb') as handle:
        return pickle.load(handle)

tokenizer = load_tokenizer()

# Load the emoji labels
@st.cache_data
def load_labels():
    with open('emoji_labels.pickle', 'rb') as handle:
        return pickle.load(handle)

emoji_labels = load_labels()

# Define the text cleaning function (must be identical to the one used in training)
stop_words = set(stopwords.words('english'))
def clean_and_process_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)

# --- Streamlit App Interface ---

st.title("🧠 AI Emoji Predictor")
st.write("A demonstration of how AI understands language and emotion.")
st.write("This tool is part of a workshop on 'Cognitive Bias in AI Recommendations'.")

# User input
user_input = st.text_input("Enter a sentence to see the predicted emoji:", "I am so happy to see you")

if st.button("Predict Emoji"):
    if user_input:
        # 1. Clean the input text
        cleaned_input = clean_and_process_text(user_input)

        # 2. Tokenize and pad the sequence
        seq = tokenizer.texts_to_sequences([cleaned_input])
        # Note: This max_len should ideally match the one from your training data.
        # You can find it in Colab with `X_train.shape[1]`. 40 is a safe default.
        max_len = 40 
        padded_seq = pad_sequences(seq, maxlen=max_len, padding='post')

        # 3. Make a prediction
        prediction = model.predict(padded_seq)
        predicted_index = np.argmax(prediction)
        predicted_emoji_class = emoji_labels[predicted_index]

        # Display the result
        st.success(f"Predicted Emoji: {predicted_emoji_class}")

    else:
        st.warning("Please enter a sentence.")