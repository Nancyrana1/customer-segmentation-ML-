from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np


# Load the pickled model
with open("customer_segmentation.pkl", "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        data = pd.read_csv(file)

        # Select the same features the model was trained on
        X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

        # Predict the clusters
        data['Cluster'] = model.predict(X)

        return data.to_html(classes='table table-striped')

if __name__ == '__main__':
    app.run(debug=True)
