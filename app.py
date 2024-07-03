from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import re
STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_length', methods=['POST'])
def get_length():
    user_input = request.form['user_input']
    predictor = pickle.load(open(r"Models/model_xgb.pkl", "rb"))
    scaler = pickle.load(open(r"Models/scaler.pkl", "rb"))
    cv = pickle.load(open(r"Models/countVectorizer.pkl", "rb"))

    predicted_sentiment = single_prediction(predictor, scaler, cv, user_input)

    return render_template('index.html', result=predicted_sentiment)

def single_prediction(predictor, scaler, cv, text_input):
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    review = " ".join(review)
    corpus.append(review)
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)[0]

    return "Positive" if y_predictions == 1 else "Negative"

if __name__ == '__main__':
    app.run(debug=True)
