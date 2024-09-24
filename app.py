from flask import Flask, request, jsonify
import numpy as np
import pickle
from flask_cors import CORS

# Load the models and scalers
model = pickle.load(open('decision_tree_model.pkl', 'rb'))
minmax_scaler = pickle.load(open('minmaxscaler.pkl', 'rb'))
standard_scaler = pickle.load(open('standard_scaler.pkl', 'rb'))  # Load the StandardScaler

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

@app.route('/')
def index():
    return "Crop Recommendation System Backend is Running"

@app.route("/predict", methods=['POST'])
def predict():
    # Collecting user inputs from the form
    N = float(request.json['Nitrogen'])
    P = float(request.json['Phosphorus'])
    K = float(request.json['Potassium'])
    temp = float(request.json['Temperature'])
    humidity = float(request.json['Humidity'])
    ph = float(request.json['Ph'])
    rainfall = float(request.json['Rainfall'])

    # Feature input list
    feature_list = [N, P, K, temp, humidity, ph, rainfall]

    # Convert to NumPy array and reshape for model prediction
    single_pred = np.array(feature_list).reshape(1, -1)

    # Choose which scaler to use (MinMaxScaler or StandardScaler)
    # Example: using MinMaxScaler for scaling features
    scaled_features = minmax_scaler.transform(single_pred)

    # Making prediction
    prediction = model.predict(scaled_features)

    # Crop Dictionary
    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
        8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
        14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
        19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }

    # Map prediction to crop
    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = {"crop": crop, "message": "{} is the best crop to be cultivated.".format(crop)}
    else:
        result = {"message": "Sorry, we could not determine the best crop with the provided data."}
    
    return jsonify(result)

# Running the app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
