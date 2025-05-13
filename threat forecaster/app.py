from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and preprocessor
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('preprocessor.pkl', 'rb') as file:
    preprocessor = pickle.load(file)

# Load train.csv to get unique values for dropdowns
train = pd.read_csv('train.csv')

# Define important columns for the form
important_cols = [
    'RealTimeProtectionState', 'NumAntivirusProductsInstalled', 'IsSystemProtected',
    'EngineVersion', 'AppVersion', 'OSVersion', 'IsSecureBootEnabled'
]

# Get unique values for dropdowns
dropdown_options = {}
for col in important_cols:
    if train[col].dtype == 'object':
        dropdown_options[col] = sorted(train[col].dropna().unique().tolist())
    else:
        dropdown_options[col] = sorted(train[col].dropna().unique().tolist())

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Collect form data
            input_data = {}
            for col in important_cols:
                input_data[col] = request.form.get(col)

            # Create a DataFrame from input data
            input_df = pd.DataFrame([input_data])

            # Convert numerical columns to appropriate type
            numerical_cols = ['RealTimeProtectionState', 'NumAntivirusProductsInstalled', 
                             'IsSystemProtected', 'IsSecureBootEnabled']
            for col in numerical_cols:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

            # Apply preprocessing
            input_processed = preprocessor.transform(input_df)

            # Make prediction
            prediction = model.predict(input_processed)[0]
            probability = model.predict_proba(input_processed)[0][1] * 100

            # Prepare Deadpool-style result message
            if prediction == 1:
                result = ("Danger alert! Your system’s more exposed than a piñata at a birthday bash! Patch it up!")
            else:
                result = ("Smooth sailing, buddy—your system’s chillin’ like a popsicle in a snowstorm. Stay frosty!")

            # Redirect to results page
            return render_template('results.html', user_inputs=input_data, 
                                 probability=probability, result=result)

        except Exception as e:
            error_message = f"Your system’s about to get roasted worse than a chimichanga left in the oven too long! Get it checked, stat! {str(e)}"
            print(error_message)  # Log to terminal
            return render_template('index.html', dropdown_options=dropdown_options, 
                                 result=error_message)

    return render_template('index.html', dropdown_options=dropdown_options, result=None)

if __name__ == '__main__':
    app.run(debug=True)