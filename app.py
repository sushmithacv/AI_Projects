from flask import Flask, request, render_template
import numpy as np
import pickle

# Load the models and scalers
model = pickle.load(open('decision_tree_model.pkl','rb'))

ms = pickle.load(open('minmaxscaler.pkl','rb'))

# Create Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    # Collecting user inputs from the form
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosphorus'])
    K = float(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['Ph'])
    rainfall = float(request.form['Rainfall'])

    # Feature input list
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    
    # Convert to NumPy array and reshape for model prediction
    single_pred = np.array(feature_list).reshape(1, -1)
    
    # Scaling features
    scaled_features = ms.transform(single_pred)  # MinMax scaling

    # Predict using pre-trained model
    prediction = model.predict(scaled_features)

    # Crop label dictionary based on dataset
    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 
                 6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 
                 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana", 
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 
                 18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans", 
                 21: "Chickpea", 22: "Coffee"}

    # Determine crop based on prediction
    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated in the given conditions.".format(crop)
    else:
        result = "Sorry, we couldn't determine the best crop for cultivation with the given data."

    # Render result in the webpage
    return render_template('index.html', result=result)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
