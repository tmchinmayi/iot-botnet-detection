from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model/botnet_model.pkl")
scaler = joblib.load("model/scaler.pkl")


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    # Read CSV file
    data = pd.read_csv(file)

    # Remove label column if present
    if "label" in data.columns:
        data = data.drop("label", axis=1)

    # Scale data
    data_scaled = scaler.transform(data)

    # Predict
    predictions = model.predict(data_scaled)

    benign = int((predictions == 0).sum())
    attack = int((predictions == 1).sum())
    total = len(predictions)

    status = "SAFE"
    if attack > benign:
        status = "BOTNET ATTACK DETECTED"

    return render_template(
        "result.html",
        total=total,
        benign=benign,
        attack=attack,
        status=status
    )


# Run server (important for Render)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)