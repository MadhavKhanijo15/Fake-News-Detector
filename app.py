
import streamlit as st
import nltk

# Download required NLTK resources
nltk.download("stopwords", quiet=True)

from src.predict import predict_news


# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)


# Application heading
st.title("🔍 Fake News Detector")

st.markdown(
    "### Verify the credibility of news using Machine Learning."
)

st.write(
    "Paste a news article below and let our trained "
    "Logistic Regression model analyse it."
)

st.divider()


# News article input
news_text = st.text_area(
    "Enter News Article",
    placeholder="Paste the news article here...",
    height=250
)


# Analyse button
if st.button("Analyse News", type="primary", use_container_width=True):

    if not news_text.strip():

        st.warning("Please enter a news article first.")

    else:

        try:
            with st.spinner("Analysing article..."):

                result = predict_news(news_text)

            prediction = result["prediction"]
            confidence = result["confidence"]

            st.divider()

            st.subheader("Analysis Result")

            if prediction == "REAL":

                st.success("Prediction: REAL NEWS")

            else:

                st.error("Prediction: FAKE NEWS")

            st.metric(
                label="Model Probability",
                value=f"{confidence}%"
            )

            st.progress(confidence / 100)

            st.caption(
                "This probability represents the model's "
                "estimated probability for its predicted class. "
                "It is not independent verification of authenticity."
            )

        except Exception as e:

            st.error(f"An error occurred: {e}")


st.divider()

st.caption(
    "Fake News Detector | Built using Python, "
    "Scikit-learn, NLTK and Streamlit"
)
