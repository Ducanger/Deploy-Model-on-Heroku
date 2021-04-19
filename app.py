from flask import Flask, render_template, request
#from vncorenlp import VnCoreNLP
import joblib
import numpy as np
import pandas as pd

#vncorenlp_file = r'VnCoreNLP-1.1.1.jar'
#vncorenlp = VnCoreNLP(vncorenlp_file)

model_svm = joblib.load('model_predict/svm.pkl')
model_lgbm = joblib.load('model_predict/lgbm.pkl')
model_catboost = joblib.load('model_predcit/catboost.pkl')
tfidf = joblib.load( 'model_predict/tfidf.pkl')

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html') 

@app.route('/predict', methods=['POST'])

def home():
    text = request.form['input']
    request_model = request.form['model']

    #listWord = vncorenlp.tokenize(text)
    #X = ' '.join([str(j) for i in listWord for j in i])
    features = tfidf.transform([text])
      
    if request_model == "2":
        model = model_lgbm
    elif request_model == "3":
        model = model_catboost
    else:
        model = model_svm

    pred = model.predict_proba(features) [:,1]
    
    return render_template('after.html', proba = pred)

if __name__ == "__main__":
    app.run(debug=True)