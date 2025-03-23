import re
import string
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS  # Import CORS

# Load dataset
df = pd.read_csv("model/balanced_b2b_emails.csv")

# Preprocess text function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

# Clean the dataset emails
df["cleaned_email"] = df["email"].apply(preprocess_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["cleaned_email"])

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Home route renders the HTML form
@app.route("/")
def home():
    return render_template("index.html")

# API route to predict spam
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_email = data.get("message", "")

    # Preprocess and vectorize
    user_email_cleaned = preprocess_text(user_email)
    user_vector = vectorizer.transform([user_email_cleaned])
    similarities = cosine_similarity(user_vector, tfidf_matrix)
    max_similarity = similarities.max()

    # Classify
    threshold = 0.30
    result = "Spam" if max_similarity > threshold else "Not Spam"

    return jsonify({
        "input": user_email,
        "classification": result,
        "similarity_score": round(float(max_similarity), 3)
    })

if __name__ == "__main__":
    app.run(debug=True)
