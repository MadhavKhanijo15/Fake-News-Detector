
import re
import html

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialize preprocessing components
stop_words = set(stopwords.words('english'))

# Preserve negation words
stop_words = stop_words - {'no', 'nor', 'not'}

stemmer = PorterStemmer()


def preprocess_text(text):

    # Convert to string and lowercase
    text = str(text).lower()

    # Decode HTML entities
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)

    # Remove unwanted characters
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenization
    tokens = text.split()

    # Stopword removal
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # Stemming
    tokens = [
        stemmer.stem(word)
        for word in tokens
    ]

    # Join tokens back into text
    return ' '.join(tokens)
