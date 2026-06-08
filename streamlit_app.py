
import streamlit as st
import pandas as pd
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

# Download NLTK data (if not already downloaded)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# Load the trained model, vectorizer, and label encoder
try:
    model = joblib.load('random_forest_model.pkl')
    vectorizer = joblib.load('count_vectorizer.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
except FileNotFoundError:
    st.error("Model, vectorizer, or label encoder files not found. Please ensure 'random_forest_model.pkl', 'count_vectorizer.pkl', and 'label_encoder.pkl' are in the same directory as this app.")
    st.stop()

# Preprocessing function (consistent with notebook preprocessing)
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    return ' '.join(filtered_tokens)

st.title('Sentiment Analysis App')
st.write('Enter a movie review to predict its sentiment (positive/negative).')

# Text input from user
user_input = st.text_area('Enter your review here:', '')

if st.button('Predict Sentiment'):
    if user_input:
        # Preprocess the input text
        processed_text = preprocess_text(user_input)

        # Transform the preprocessed text using the loaded vectorizer
        text_vectorized = vectorizer.transform([processed_text])

        # Make prediction
        prediction_encoded = model.predict(text_vectorized)

        # Decode the prediction
        prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]

        st.success(f'Predicted Sentiment: **{prediction_label.capitalize()}**')
    else:
        st.warning('Please enter some text to analyze.')
