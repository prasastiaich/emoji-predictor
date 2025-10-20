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

# Load the mapping from number to emoji character
@st.cache_data
def load_mapping():
    # --- FIX: Read the CSV using its header row (header=0) ---
    # This correctly uses the column names from your file.
    mapping_df = pd.read_csv('Mapping.csv', header=0)
    
    # Drop any rows that might be missing values in the essential columns.
    mapping_df.dropna(subset=['number', 'emoticons'], inplace=True)
    
    # Now, it's safe to convert the 'number' column to the integer type.
    mapping_df['number'] = mapping_df['number'].astype(int)
    
    # Create the dictionary from the 'number' and 'emoticons' columns.
    return pd.Series(mapping_df.emoticons.values, index=mapping_df.number).to_dict()

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

user_input = st.text_input("Enter a sentence to see the predicted emoji:", "I am so happy to see you")

if st.button("Predict Emoji"):
    if user_input:
        cleaned_input = clean_and_process_text(user_input)
        seq = tokenizer.texts_to_sequences([cleaned_input])
        max_len = 40 
        padded_seq = pad_sequences(seq, maxlen=max_len, padding='post')
        
        prediction = model.predict(padded_seq)
        predicted_index = np.argmax(prediction)
        
        predicted_class_number = emoji_labels[predicted_index]

        # Convert the predicted number to a standard Python integer to ensure the lookup works.
        final_emoji = emoji_map.get(int(predicted_class_number), "🤔")
        
        st.success(f"Predicted Emoji: {final_emoji}")

    else:
        st.warning("Please enter a sentence.")

