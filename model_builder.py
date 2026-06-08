
import pandas as pd
import re
import string
import nltk
import joblib
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Ensure NLTK data is downloaded
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# Preprocessing function (consistent with notebook preprocessing)
def preprocess_text_and_remove_duplicates(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    unique_words = list(set(filtered_tokens))
    return ' '.join(unique_words)

def build_and_save_model(csv_path='IMDB_Dataset.csv', model_name='random_forest_model.pkl',
                         vectorizer_name='count_vectorizer.pkl', label_encoder_name='label_encoder.pkl'):
    # 1. Load the data
    df = pd.read_csv(csv_path, engine='python', on_bad_lines='skip')

    # 2. Preprocess data
    df['cleaned_review'] = df['review'].apply(preprocess_text_and_remove_duplicates)

    # 3. Split data
    X = df['cleaned_review']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Initialize and fit CountVectorizer
    count_vectorizer = CountVectorizer(max_features=100000)
    X_train_count = count_vectorizer.fit_transform(X_train)
    X_test_count = count_vectorizer.transform(X_test)

    # 5. Initialize and fit LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train) # Fit on training labels

    # 6. Train the best performing model (Random Forest from previous tuning)
    # These parameters are based on the (hypothetical) best found during tuning
    best_rf_model = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=1, min_samples_split=2, random_state=42)
    best_rf_model.fit(X_train_count, y_train)

    # 7. Save the trained model, vectorizer, and label encoder
    joblib.dump(best_rf_model, model_name)
    joblib.dump(count_vectorizer, vectorizer_name)
    joblib.dump(label_encoder, label_encoder_name)

    print(f"Model '{model_name}', vectorizer '{vectorizer_name}', and label encoder '{label_encoder_name}' saved successfully.")

if __name__ == '__main__':
    build_and_save_model()
