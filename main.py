import os
import pickle
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK stopwords if not already available
nltk.download("stopwords")

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and tokenizer
model = load_model("lstm_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


# Text preprocessing function
def clean_text(text):
    """Cleans text by lowercasing, removing stopwords, and applying stemming."""
    ps = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)  # Remove non-alphabetic characters
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]

    return " ".join(words)


# Prediction function
def predict_sentiment(text):
    """Predicts sentiment for a given text."""
    text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    prediction = model.predict(padded_sequence)[0][0]

    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return sentiment, float(prediction)


# Home and prediction route
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None

    if request.method == "POST":
        text = request.form.get("text")
        if text:
            result, confidence = predict_sentiment(text)

    return render_template("index.html", result=result, confidence=round(confidence,2))


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
