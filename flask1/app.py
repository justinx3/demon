from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    val1 = request.form['SL']
    val2 = request.form['SW']
    val3 = request.form['PL']
    val4 = request.form['PW']
    arr = np.array([val1, val2, val3, val4])
    arr = arr.astype(np.float64)
    pred = model.predict([arr])
    pred=int(pred)
    if pred==0:
        pred='Iris-setosa'
    elif pred==1:
        pred='Iris-versicolor'
    else:
        pred='Iris-virginica'

    return render_template('index.html', data=(pred))


if __name__ == '__main__':
    app.run(debug=True)
