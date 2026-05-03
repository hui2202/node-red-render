from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the saved model
model = joblib.load('traffic_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from Node-RED
        data = request.get_json()
        
        # Prepare input for model
        features = np.array([[
            float(data['temp']), 
            float(data['rain_1h']), 
            int(data['hour']), 
            int(data['is_peak']), 
            int(data['day_of_week'])
        ]])
        
        # Predict result
        prediction = model.predict(features)[0]
        
        # Return to Node-RED (0=Low, 1=Med, 2=High)
        return jsonify({'traffic_level': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Codespaces usually uses port 5000
    app.run(host='0.0.0.0', port=5000)