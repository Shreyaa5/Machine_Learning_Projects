from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name, static_url_path='/static')

# Load the model from the pickle file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def show_form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the form data from the request
        time = int(request.form['time'])
        v1 = float(request.form['v1'])
        v2 = float(request.form['v2'])
        v3 = float(request.form['v3'])
        v4 = float(request.form['v4'])
        v5 = float(request.form['v5'])
        v6 = float(request.form['v6'])
        v7 = float(request.form['v7'])
        v8 = float(request.form['v8'])
        v9 = float(request.form['v9'])
        v10 = float(request.form['v10'])
        v11 = float(request.form['v11'])
        v12 = float(request.form['v12'])
        v13 = float(request.form['v13'])
        v14 = float(request.form['v14'])
        v15 = float(request.form['v15'])
        v16 = float(request.form['v16'])
        v17 = float(request.form['v17'])
        v18 = float(request.form['v18'])
        v19 = float(request.form['v19'])
        v20 = float(request.form['v20'])
        v21 = float(request.form['v21'])
        v22 = float(request.form['v22'])
        v23 = float(request.form['v23'])
        v24 = float(request.form['v24'])
        v25 = float(request.form['v25'])
        v26 = float(request.form['v26'])
        v27 = float(request.form['v27'])
        v28 = float(request.form['v28'])
        amount = float(request.form['amount'])

        # Process the data and make the prediction using the loaded model
        data = {
            "Time": time,
            "V1": v1,
            "V2": v2,
            "V3": v3,
            "V4": v4,
            "V5": v5,
            "V6": v6,
            "V7": v7,
            "V8": v8,
            "V9": v9,
            "V10": v10,
            "V11": v11,
            "V12": v12,
            "V13": v13,
            "V14": v14,
            "V15": v15,
            "V16": v16,
            "V17": v17,
            "V18": v18,
            "V19": v19,
            "V20": v20,
            "V21": v21,
            "V22": v22,
            "V23": v23,
            "V24": v24,
            "V25": v25,
            "V26": v26,
            "V27": v27,
            "V28": v28,
            "Amount": amount
        }

        # Convert the dictionary to a DataFrame and reshape it for prediction
        df = pd.DataFrame([data])
        first_element = df.iloc[0]
        first_element_array = np.array(first_element)
        first_element_reshaped = first_element_array.reshape(1, -1)

        # Make the prediction
        prediction = model.predict(first_element_reshaped)
        prediction_value = "Safe" if prediction == 0 else "Not Safe"

        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction_value})
    except Exception as e:
        return jsonify({'error': str(e})

if __name__ == '__main__':
    app.run(debug=True)
