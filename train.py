from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pickle
import numpy as np

# Sample data (replace later with your dataset)
texts = [
    "This is written by a human",
    "AI generates text automatically",
    "Humans write with emotions",
    "AI produces structured content"
]

labels = [0, 1, 0, 1]

# TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(texts).toarray()

# SAME FEATURE FUNCTION AS app.py
def extract_features(text):
    words = text.split()
    sentences = text.split('.')

    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    sentence_length = len(words) / len(sentences) if sentences else 0
    unique_ratio = len(set(words)) / len(words) if words else 0

    return [
        len(text),
        text.count(','),
        text.count('.'),
        avg_word_len,
        sentence_length,
        unique_ratio
    ]

# Apply extra features
extra = np.array([extract_features(t) for t in texts])

# COMBINE (IMPORTANT 🔥)
X_final = np.hstack((X, extra))

# Train model
model = LinearSVC()
model.fit(X_final, labels)

# Save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model retrained with correct features!")