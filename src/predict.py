
import joblib
from pathlib import Path

from src.preprocessing import preprocess_text


# Locate project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load saved ML artifacts
MODEL_PATH = BASE_DIR / "models" / "logistic_regression.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"

logistic_model = joblib.load(MODEL_PATH)
tfidf = joblib.load(VECTORIZER_PATH)


def predict_news(news_text):

    # Validate input
    if not news_text or not news_text.strip():
        raise ValueError("Please provide a valid news article.")

    # Step 1: Preprocess incoming article
    processed_text = preprocess_text(news_text)

    # Step 2: Convert text into TF-IDF features
    text_vector = tfidf.transform([processed_text])

    # Step 3: Predict using Logistic Regression
    prediction = logistic_model.predict(text_vector)[0]

    # Step 4: Get model probability
    probabilities = logistic_model.predict_proba(text_vector)[0]

    confidence = probabilities[list(logistic_model.classes_).index(prediction)]

    # Step 5: Map numerical prediction to label
    label = "REAL" if prediction == 1 else "FAKE"

    return {
        "prediction": label,
        "confidence": round(float(confidence) * 100, 2)
    }
