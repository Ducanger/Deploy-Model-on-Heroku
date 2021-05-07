from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

model_svm = joblib.load('model_predict/svm.pkl')
model_lgbm = joblib.load('model_predict/lgbm.pkl')
model_catboost = joblib.load('model_predict/catboost.pkl')
tfidf = joblib.load('model_predict/tfidf.pkl')
word_tokenize = joblib.load('model_predict/underthesea.pkl')

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html') 

@app.route('/predict', methods=['POST'])

def home():
    text = request.form['input']
    request_model = request.form['model']

    X = word_tokenize(text, format="text")
    features = tfidf.transform([X])
      
    if request_model == "2":
        model = model_lgbm
    elif request_model == "3":
        model = model_catboost
    else:
        model = model_svm

    pred = model.predict_proba(features) [:,1]

    return render_template('after.html', proba = pred)

if __name__ == "__main__":
    app.run(debug=False)
