# Fake-News-Detector

An NLP-based Fake News Detection system that classifies news articles as **REAL** or **FAKE** using traditional Machine Learning algorithms and TF-IDF text vectorization.

The project uses the WELFake dataset and implements two classification models: Logistic Regression and Multinomial Naive Bayes.

## Project Overview

The objective of this project is to explore how Natural Language Processing (NLP) and Machine Learning can be used to identify patterns in news articles and classify them as real or fake.

The system takes a news article's title and content as input, processes the text, extracts numerical features using TF-IDF, and generates a prediction using a trained classification model.

## Tech Stack

- **Language:** Python
- **Machine Learning:** Scikit-learn
- **NLP & Text Processing:** NLTK / Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Vectorization:** TF-IDF
- **Interface:** Streamlit
- **Development:** Google Colab, VS Code

## Dataset

This project uses the **WELFake Dataset**, containing 72,134 news articles.

Dataset distribution:
- Real News: 35,028
- Fake News: 37,106

The dataset contains the following relevant fields:

| Feature | Description |
|---|---|
| Title | Headline of the news article |
| Text | Main content of the article |
| Label | Classification label (0 = Fake, 1 = Real) |

**Dataset source:** [WELFake Dataset – Zenodo](https://zenodo.org/records/4561253)

Download the dataset separately and place it in the appropriate dataset directory if you wish to reproduce the training process.

## Project Workflow

1. **Data Loading:** Load the WELFake dataset and inspect its structure.
2. **Preprocessing:** Handle missing values and prepare the textual data.
3. **Feature Engineering:** Combine the title and article content.
4. **Text Vectorization:** Convert text into numerical features using TF-IDF.
5. **Model Training:** Train Logistic Regression and Multinomial Naive Bayes classifiers.
6. **Model Evaluation:** Evaluate performance using accuracy, precision, recall, F1-score and confusion matrices.
7. **Prediction:** Use the saved vectorizer and trained model to classify new article inputs.

## Machine Learning Models

Two supervised classification algorithms were implemented:

### 1. Logistic Regression
A linear classification algorithm used to distinguish between real and fake news based on TF-IDF features.

### 2. Multinomial Naive Bayes
A probabilistic classification algorithm commonly used for text classification tasks.

### TF-IDF Configuration

- Maximum Features: 50,000
- N-gram Range: (1, 2)
- Features: Unigrams and Bigrams

### Training Configuration

- Training Set: 80%
- Testing Set: 20%
- Splitting Strategy: Stratified Train-Test Split

## Model Evaluation

Logistic Regression achieved the following results on the held-out test set:

| Metric | Score |
|---|---:|
| Accuracy | 96.09% |
| Precision | 96.07% |
| Recall | 95.16% |
| F1-Score | 95.61% |

Both Logistic Regression and Multinomial Naive Bayes were trained and evaluated as part of the project.

## Application

The project includes a Streamlit interface where users can enter a news article and receive a classification prediction.

The application displays:
- Predicted class (REAL / FAKE)
- Model probability associated with the prediction

## Project Structure

```text
Fake-News-Detection/
│
├── app.py
│
├── src/
│   ├── preprocessing.py
│   └── prediction.py
│
├── model/
│   └── saved model and vectorizer file
│
├── dataset/
│   └── dataset files (not included , download using link provided)
│
├── requirements.txt
└── README.md
```

*Note: The structure above represents the logical organization of the project. Adjust individual filenames to match the actual repository.*

## Installation and Execution

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd Fake-News-Detector
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
```

Activate it:

Windows:
```bash
venv\Scripts\activate
```

Linux / macOS:
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

**Note:** The trained model and vectorizer files must be present in the expected `model/` directory. Download the WELFake dataset separately only if you intend to reproduce model training.

## Limitations

- The model learns statistical and linguistic patterns from the WELFake training data.
- Its performance on the dataset's test split does not guarantee equivalent performance on newer or external news sources.
- Model probability is not a verified measure of factual correctness.
- The system does not independently verify claims against authoritative sources.
- Biases, outdated information and distribution shifts in the training data may affect predictions.

This project is an educational Machine Learning implementation and should not be treated as an independent fact-checking system.

## Future Improvements

- Experiment with deep learning architectures.
- Explore transformer-based NLP models such as BERT.
- Evaluate generalization on independently verified external datasets.
- Integrate evidence retrieval and source verification.

## Acknowledgements

- WELFake Dataset by Verma et al.
- Scikit-learn documentation
- Streamlit documentation

---

## Output Snippets
<img width="1679" height="912" alt="image" src="https://github.com/user-attachments/assets/2116af7e-c0b1-4362-8b28-c01c06f60d8d" />

