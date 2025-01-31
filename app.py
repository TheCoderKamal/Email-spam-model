from flask import Flask, render_template, request, jsonify
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os

# Load the model and feature extraction using absolute paths
model = joblib.load(os.path.abspath('./model/spam_ham_model.pkl'))
feature_extraction = joblib.load(os.path.abspath('./model/vectorizer.pkl'))

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.ejs')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get JSON input
        data = request.json
        input_message = data['message']

        # Transform the input message using the fitted TfidfVectorizer
        input_features = feature_extraction.transform([input_message])

        # Predict using the loaded model
        prediction = model.predict(input_features)

        # Map prediction result
        result = 'Ham' if prediction[0] == 1 else 'Spam'

        return jsonify({"message": input_message, "prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
