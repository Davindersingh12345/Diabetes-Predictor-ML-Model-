from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    # Base numerical inputs
    Age = float(request.form['Age'])
    BMI = float(request.form['BMI'])

    # Direct dummy variables
    Physical = int(request.form['Physical_Activity'])
    Smoked = int(request.form['Smoked'])
    Gender = int(request.form['Gender'])
    HighBP = int(request.form['HighBP'])
    

    # General Health → one-hot encode into 4 features
    GH = request.form["GeneralHealth"]

    GH_Fair = 1 if GH == "Fair" else 0
    GH_Good = 1 if GH == "Good" else 0
    GH_Poor = 1 if GH == "Poor" else 0
    GH_VeryGood = 1 if GH == "VeryGood" else 0

    # FINAL FEATURE VECTOR (must match EXACT order from your X.columns list)
    data = np.array([[
        Age,
        BMI,
        Physical,
        Smoked,
        GH_Fair,
        GH_Good,
        GH_Poor,
        GH_VeryGood,
        Gender,
        HighBP
    ]])

    # Scale inputs
    data_scaled = scaler.transform(data)

    # Predict
    pred = model.predict(data_scaled)[0]
    result = "Diabetic" if pred == 1 else "Non-Diabetic"

    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)
