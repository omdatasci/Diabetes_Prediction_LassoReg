import pandas as pd
import numpy as np
import pickle

from flask import Flask, render_template, request
app = Flask(__name__)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = {
        'age' : float(request.form['Age']),
        'sex' : float(request.form['Sex']),
        'bmi' : float(request.form['Bmi']),
        'bp' : float(request.form['Bp']),
        's1' : float(request.form['S1']),
        's2' : float(request.form['S2']),
        's3' : float(request.form['S3']),
        's4' : float(request.form['S4']),
        's5' : float(request.form['S5']),
        's6' : float(request.form['S6']),
        }

    data_df = pd.DataFrame([data])
    prediction = model.predict(data_df)[0]

    return render_template('result.html', predictions=prediction)


if(__name__) == '__main__':
    app.run(debug=True)
