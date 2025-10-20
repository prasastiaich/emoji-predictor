import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd

# --- Pre-load necessary items ---
# Download stopwords if not already present
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# Load the trained model
@st.cache_resource
def load_model():
    # Make sure the path to your model file is correct
    return tf.keras.models.load_model('emoji_predictor_model.keras')

model = load_model()

# Load the tokenizer
@st.cache_data
def load_tokenizer():
    # Make sure the path to your tokenizer file is correct
    with open('tokenizer.pickle', 'rb') as handle:
        return pickle.load(handle)

tokenizer = load_tokenizer()

# Load the emoji labels (which are numbers)
@st.cache_data
def load_labels():
    # Make sure the path to your labels file is correct
    with open('emoji_labels.pickle', 'rb') as handle:
        return pickle.load(handle)

emoji_labels = load_labels()

# Load the mapping from number to emoji character
@st.cache_data
def load_mapping():
    # Make sure the path to your Mapping.csv file is correct
    mapping_df = pd.read_csv('Mapping.csv', header=None, usecols=[0, 1])
    mapping_df.columns = ['number', 'emoji']
    # Create a dictionary for easy lookup: {0: '❤️', 1: '😊', ...}
    return pd.Series(mapping_df.emoji.values, index=mapping_df.number).to_dict()

emoji_map = load_mapping()


# Define the text cleaning function
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
        max_len = 40 
        padded_seq = pad_sequences(seq, maxlen=max_len, padding='post')
        
        # 3. Make a prediction
        prediction = model.predict(padded_seq)
        predicted_index = np.argmax(prediction)
        
        # 4. Translate the prediction
        # First, get the predicted class NUMBER (e.g., 9)
        predicted_class_number = emoji_labels[predicted_index]
        
        # Then, use the map to get the EMOJI (e.g., '❤️')
        # Try to find the emoji. If not found, use '🤔' as a default.
        # --- FIX: This line was un-indented. It now correctly sits inside this 'if' block. ---
        final_emoji = emoji_map.get(predicted_class_number, "🤔")
        
        # 5. Display the final emoji result
        st.success(f"Predicted Emoji: {final_emoji}")

    else:
        st.warning("Please enter a sentence.")
