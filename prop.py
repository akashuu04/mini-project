import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import re

# ----------------------------
# Load datasets
# ----------------------------
fake_df = pd.read_csv(r"C:\Users\Akash basavaraja\Desktop\END\end\datasets\fake-and-real-news-dataset\Fake.csv")
true_df = pd.read_csv(r"C:\Users\Akash basavaraja\Desktop\END\end\datasets\fake-and-real-news-dataset\True.csv")
propaganda_df = pd.read_csv(r"C:\Users\Akash basavaraja\Desktop\END\end\datasets\fake-and-real-news-dataset\z_combined.csv")

# ----------------------------
# Labels
# ----------------------------
fake_df['label'] = 'not_propaganda'
true_df['label'] = 'not_propaganda'
propaganda_df['label'] = 'propaganda'

# ----------------------------
# Combine datasets
# ----------------------------
prop_df = pd.concat([fake_df, true_df, propaganda_df], ignore_index=True)
prop_df = prop_df[['text', 'label']]
prop_df['label_enc'] = prop_df['label'].map({'not_propaganda': 0, 'propaganda': 1})

# ----------------------------
# Clean text (remove sample IDs)
# ----------------------------
def clean_text(text):
    text = re.sub(r"\[Sample \d+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

prop_df['text'] = prop_df['text'].apply(clean_text)

# ----------------------------
# Balance dataset by oversampling minority class
# ----------------------------
propaganda = prop_df[prop_df['label_enc'] == 1]
not_propaganda = prop_df[prop_df['label_enc'] == 0]

if len(propaganda) > len(not_propaganda):
    not_propaganda_upsampled = resample(not_propaganda,
                                        replace=True,
                                        n_samples=len(propaganda),
                                        random_state=42)
    balanced_df = pd.concat([propaganda, not_propaganda_upsampled])
else:
    propaganda_upsampled = resample(propaganda,
                                    replace=True,
                                    n_samples=len(not_propaganda),
                                    random_state=42)
    balanced_df = pd.concat([not_propaganda, propaganda_upsampled])

balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# ----------------------------
# Split train/test
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    balanced_df['text'],
    balanced_df['label_enc'],
    test_size=0.2,
    random_state=42,
    stratify=balanced_df['label_enc']
)

# ----------------------------
# TF-IDF Vectorization (unigrams + bigrams)
# ----------------------------
vectorizer_prop = TfidfVectorizer(max_features=7000, ngram_range=(1,2))
X_train_vec = vectorizer_prop.fit_transform(X_train)
X_test_vec = vectorizer_prop.transform(X_test)

# ----------------------------
# Train Naive Bayes
# ----------------------------
model_prop = MultinomialNB()
model_prop.fit(X_train_vec, y_train)

# ----------------------------
# Save model and vectorizer
# ----------------------------
joblib.dump(model_prop, r"C:\Users\Akash basavaraja\Desktop\END\end\propaganda_model.pkl")
joblib.dump(vectorizer_prop, r"C:\Users\Akash basavaraja\Desktop\END\end\propaganda_vectorizer.pkl")

print("✅ Stronger propaganda model saved!")

# ----------------------------
# Evaluate
# ----------------------------
y_pred = model_prop.predict(X_test_vec)
print("\nClassification Report (threshold 0.5):")
print(classification_report(y_test, y_pred, target_names=['not_propaganda', 'propaganda']))

probs = model_prop.predict_proba(X_test_vec)[:,1]
y_pred_thresh = (probs >= 0.4).astype(int)
print("\nClassification Report (threshold 0.4, favor propaganda):")
print(classification_report(y_test, y_pred_thresh, target_names=['not_propaganda', 'propaganda']))
