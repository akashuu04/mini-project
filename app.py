import streamlit as st
import joblib
from datetime import date
from model import (
    predict_fake_real, predict_propaganda, sentiment_analysis,
    transparency_score, fake_real_explanation, domain_recommendation
)

# ----------------------------
# Load Models
# ----------------------------
fake_real_model = joblib.load("fake_real_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ----------------------------
# Streamlit Frontend
# ----------------------------
st.title("News Analyzer: Fake/Real + Propaganda + Sentiment + Transparency + AI Explanation")

article_text = st.text_area("Enter news article text here:")
url_input = st.text_input("Enter article URL:")
author_input = st.text_input("Enter Author Name (optional):")
# Calendar date picker
date_input = st.date_input("Select Date (optional)", value=None)
if date_input is not None:
    date_input = date_input.strftime("%Y-%m-%d")

if st.button("Analyze"):
    if article_text.strip() == "":
        st.warning("Please enter the news article!")
    else:
        # ----------------------------
        # Fake vs Real Prediction
        # ----------------------------
        fake_real_pred = predict_fake_real(article_text, fake_real_model, vectorizer, label_encoder)
        st.subheader("Fake vs Real Detection:")
        color = "green" if fake_real_pred.lower() == "real" else "red"
        st.markdown(f"Prediction: <span style='color:{color}; font-weight:bold'>{fake_real_pred}</span>", unsafe_allow_html=True)

        # ----------------------------
        # Propaganda Detection
        # ----------------------------
        propaganda_pred = predict_propaganda(article_text)
        st.subheader("Propaganda Detection:")
        color = "yellow" if propaganda_pred.lower() == "propaganda" else "orange"
        st.markdown(f"This article is <span style='color:{color}; font-weight:bold'>{propaganda_pred}</span>", unsafe_allow_html=True)

        # ----------------------------
        # Sentiment Analysis
        # ----------------------------
        sentiment_result = sentiment_analysis(article_text)
        st.subheader("Sentiment Analysis:")
        if sentiment_result.lower() == "positive":
            color = "green"
        elif sentiment_result.lower() == "negative":
            color = "red"
        else:
            color = "blue"
        st.markdown(f"Sentiment: <span style='color:{color}; font-weight:bold'>{sentiment_result}</span>", unsafe_allow_html=True)

        # ----------------------------
        # Transparency Score
        # ----------------------------
        st.subheader("Transparency Score:")
        score, breakdown = transparency_score(article_text, url_input, author_input, date_input)
        st.write(f"**Score:** {score}")
        with st.expander("See Breakdown"):
            for item in breakdown:
                st.write("- " + item)

        # ----------------------------
        # Explanation (Professional 4-5 sentence)
        # ----------------------------
        st.subheader("Explanation (Why Fake/Real):")
        explanation = fake_real_explanation(
            article_text, fake_real_model, vectorizer, label_encoder,
            url=url_input, author=author_input, date=date_input
        )
        st.write(explanation)

        # ----------------------------
        # Domain Recommendation
        # ----------------------------
        st.subheader("Domain Recommendation:")
        recommendation = domain_recommendation(fake_real_pred, url_input)
        st.write(recommendation)
