import joblib
from textblob import TextBlob

# ----------------------------
# Load propaganda model
# ----------------------------
propaganda_model = joblib.load(r"C:\Users\Akash basavaraja\Desktop\END\end\propaganda_model.pkl")
propaganda_vectorizer = joblib.load(r"C:\Users\Akash basavaraja\Desktop\END\end\propaganda_vectorizer.pkl")

# ----------------------------
# Fake/Real Prediction
# ----------------------------
def predict_fake_real(text, model, vectorizer, label_encoder):
    X_vec = vectorizer.transform([text])
    pred_enc = model.predict(X_vec)[0]
    pred_label = label_encoder.inverse_transform([pred_enc])[0]
    return pred_label

# ----------------------------
# Propaganda Detection
# ----------------------------
def predict_propaganda(text, model=propaganda_model, vectorizer=propaganda_vectorizer):
    X_vec = vectorizer.transform([text])
    pred = model.predict(X_vec)[0]
    return "Propaganda" if pred == 1 else "Not Propaganda"

# ----------------------------
# Sentiment Analysis
# ----------------------------
def sentiment_analysis(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# ----------------------------
# Transparency Score
# ----------------------------
def transparency_score(text, url=None, author=None, date=None):
    score = 0.0
    breakdown = []

    # Author
    if author and author.strip() != "":
        score += 0.2
        breakdown.append("+0.2 Author present")
    else:
        breakdown.append("0 Author missing")

    # Date
    if date and date.strip() != "":
        score += 0.2
        breakdown.append("+0.2 Date present")
    else:
        breakdown.append("0 Date missing")

    # Domain trust
    trusted_domains = [
        "bbc.com", "nytimes.com", "reuters.com", "theguardian.com",
        "cnn.com", "aljazeera.com", "indiatoday.in", "ndtv.com"
    ]
    if url:
        if any(domain in url for domain in trusted_domains):
            score += 0.3
            breakdown.append("+0.3 Trusted domain")
        else:
            score -= 0.1
            breakdown.append("-0.1 Untrusted/unknown domain")
    else:
        breakdown.append("0 No URL provided")

    score = max(0.0, min(1.0, score))
    return round(score, 2), breakdown

# ----------------------------
# Fake/Real Professional Explanation (4-5 sentence sentiment)
# ----------------------------
def fake_real_explanation(text, model, vectorizer, label_encoder, url=None, author=None, date=None):
    """
    Generates a professional explanation for Fake/Real prediction,
    including detailed 4–5 sentence sentiment analysis and transparency score.
    """
    # Prediction
    X_vec = vectorizer.transform([text])
    pred_enc = model.predict(X_vec)[0]
    pred_label = label_encoder.inverse_transform([pred_enc])[0]

    # Sentiment analysis with 4-5 sentence professional description
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if pred_label.lower() == "fake":
        sentiment_desc = (
            "The tone of this article is strongly negative. "
            "It conveys alarm and fear, emphasizing urgent threats and potential chaos. "
            "The language is emotionally charged, aiming to provoke concern and immediate reaction from readers. "
            "There is clear use of manipulative phrasing, exaggeration, and biased statements that heighten tension. "
            "Overall, the sentiment reinforces the sensationalist and misleading nature of the content."
        )
        base_explanation = (
            "The model predicts this article as **Fake**. "
            "It uses sensationalist language, exaggerated warnings, and manipulative phrasing designed to provoke strong reactions. "
        )
    else:
        sentiment_desc = (
            "The tone of this article is neutral and factual. "
            "It presents information in a structured and balanced manner without attempting to manipulate the reader’s emotions. "
            "The language is measured, providing evidence and context for each statement. "
            "There is no exaggeration or bias, and the reporting style encourages informed understanding. "
            "Overall, the sentiment reflects credibility and professional journalistic standards."
        )
        base_explanation = (
            "The model predicts this article as **Real**. "
            "The content is factual, structured, and neutral, presenting balanced information. "
        )

    # Transparency
    score, breakdown = transparency_score(text, url, author, date)
    transparency_str = f"Transparency score: {score} — " + "; ".join(breakdown)

    # Combine final explanation
    explanation = f"{base_explanation}{sentiment_desc} {transparency_str}."

    return explanation

# ----------------------------
# Domain Recommendation
# ----------------------------
def domain_recommendation(fake_real_pred, url):
    trusted_domains = [
        "bbc.com", "nytimes.com", "reuters.com", "theguardian.com",
        "cnn.com", "aljazeera.com", "indiatoday.in", "ndtv.com"
    ]
    if url:
        if any(domain in url for domain in trusted_domains):
            if fake_real_pred.lower() == "real":
                return "✅ Domain is trusted and news is Real. Likely trustworthy."
            else:
                return "⚠️ Domain is trusted but news is Fake. Be cautious!"
        else:
            return "❌ Domain is not trusted. Do not fully trust this news article."
    else:
        return "ℹ️ No URL provided. Cannot assess domain trust."
