from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load('house_price_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'HEAD'])
def predict():
    area = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    location = request.form['location']
    age = int(request.form['age'])
    
    # Dummy encoding (same as training)
    loc_downtown = 1 if location == 'Downtown' else 0
    loc_suburb = 1 if location == 'Suburb' else 0
    
    features = np.array([[area, bedrooms, age, loc_downtown, loc_suburb]])
    prediction = model.predict(features)[0]
    
    return render_template('index.html', prediction_text=f"Estimated House Price: ${prediction:,.2f}")

if __name__ == "__main__":
    app.run(debug=True)
