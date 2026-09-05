import re
import joblib
import streamlit as st

model = joblib.load("data/sentiment_model.pkl")
tfidf = joblib.load("data/tfidf_vectorizer.pkl")

#function to clean the text

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

st.title("Sentiment Analysis App")
review= st.text_area("Enter Your Review")

if st.button("Predict"):

    if review.strip()=="":
        st.warning("Please Enter A Review!!")

    else:
        cleaned_review = clean_text(review)

        review_vector = tfidf.transform([cleaned_review])

        prediction = model.predict(review_vector)[0]

        if prediction == "Positive":
            st.success("Positive")
        else:
            st.error("Negative")