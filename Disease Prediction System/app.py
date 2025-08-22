from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__, static_url_path='/static')

# Load the model from the pickle file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')  
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the form data from the request
        itching = int(request.form['itching'])
        skin_rash = int(request.form['skin_rash'])
        nodal_skin_eruptions = int(request.form['nodal_skin_eruptions'])
        continuous_sneezing = int(request.form['continuous_sneezing'])
        shivering = int(request.form['shivering'])
        small_dents_in_nails = int(request.form['small_dents_in_nails'])
        inflammatory_nails = int(request.form['inflammatory_nails'])
        blister = int(request.form['blister'])
        red_sore_around_nose = int(request.form['red_sore_around_nose'])
        yellow_crust_ooze = int(request.form['yellow_crust_ooze'])

        # Process the data and make the prediction using the loaded model
        data = {
            "itching": itching,
            "skin_rash": skin_rash,
            "nodal_skin_eruptions": nodal_skin_eruptions,
            "continuous_sneezing": continuous_sneezing,
            "shivering": shivering,
            "small_dents_in_nails": small_dents_in_nails,
            "inflammatory_nails": inflammatory_nails,
            "blister": blister,
            "red_sore_around_nose": red_sore_around_nose,
            "yellow_crust_ooze": yellow_crust_ooze
        }

        # Convert the dictionary to a DataFrame and reshape it for prediction
        df = pd.DataFrame([data])
        first_element = df.iloc[0]
        first_element_array = np.array(first_element)
        first_element_reshaped = first_element_array.reshape(1, -1)

        # Make the prediction
        prediction = model.predict(first_element_reshaped)
        prediction_value = "Positive" if prediction == 1 else "Negative"  # Change labels based on model predictions

        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction_value})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
