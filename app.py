import pickle
import warnings
import sklearn
from flask import Flask, request, jsonify
warnings.filterwarnings('ignore')

# Creating a Flask app
app = Flask(__name__)

# Loading the machine learning model from a pickle file
model, word_vectorizer = pickle.load(open("model.pkl", 'rb'))

@app.route("/predict/", methods=["GET"])
def display():
    return "Server running..."

# Define a route for making predictions
@app.route("/predict/", methods=["POST"])
def predict():
    # Get JSON data from the request
    data = dict(request.json)
    word_features = word_vectorizer.transform([data['categories']])
    predicted_probabilities = model.predict_proba(word_features)
    top_3_indices = predicted_probabilities.argsort(axis=1)[:, -3:][:, ::-1]
    top_3_categories = model.classes_[top_3_indices]
    top_3_probabilities = predicted_probabilities[0, top_3_indices[0]]

    result = []
    for j in range(3):
        result.append({"Category": top_3_categories[0, j], "Probability": top_3_probabilities[j]})

    # Return the predictions as a JSON response
    return jsonify({"result": result})

# Run the Flask app when this script is executed
if __name__ == "__main__":
    app.run()