
import pandas as pd
from flask import Flask, render_template, request
from naive_bayes_model import BayesModel

app = Flask(__name__)

# Load your pre-trained machine learning model here
model = BayesModel()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get user input from the form
        input_data = request.form.get("input")

        # Preprocess the input data
        X = [[float(x.strip()) for x in input_data.split(",")]]
        
        prediction = model.predict(X)

        # Format the prediction for display
        predicted_class = prediction[0]
        outcome = ""
        if (predicted_class == 0):
            outcome = "The patient is predicted to survive the 6 months following heart failure."
        else:
            outcome = "The patient is predicted to die within 6 months of the initial heart failure."

        return render_template("results.html", prediction=outcome)

    else:
        return "Something went wrong. Please try again."

if __name__ == "__main__":
    app.run(debug=True)
