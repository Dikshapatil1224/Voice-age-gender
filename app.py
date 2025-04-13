# app.py
from flask import Flask, request, render_template
from utils import extract_features_from_csv_style
import joblib
import os
import soundfile as sf

app = Flask(__name__)
model = joblib.load("model/voice_gender_age_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        file = request.files["audio"]
        filepath = os.path.join("uploads", file.filename)
        file.save(filepath)
        features = extract_features_from_csv_style(filepath).reshape(1, -1)
        prediction = model.predict(features)[0]
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)