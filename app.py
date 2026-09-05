import re
import joblib
import streamlit as st

model = joblib.load("data/sentiment_model.pkl")
tfidf = joblib.load("data/tfidf_vectorizer.pkl")

