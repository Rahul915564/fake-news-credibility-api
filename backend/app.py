import os
import joblib
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ---------- ABSOLUTE PATH (CORRECT) ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "ml", "model_v2.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "..", "ml", "vectorizer_v2.pkl")

MODEL_PATH = os.path.abspath(MODEL_PATH)
VECTORIZER_PATH = os.path.abspath(VECTORIZER_PATH)

print("MODEL PATH:", MODEL_PATH)
print("VECTORIZER PATH:", VECTORIZER_PATH)

# ---------- LOAD MODEL ----------
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def get_top_words(text, vectorizer, model, top_n=5):
    vec = vectorizer.transform([text])

    feature_names = vectorizer.get_feature_names_out()
    coef = model.coef_[0]

    indices = vec.nonzero()[1]

    word_scores = [(feature_names[i], coef[i]) for i in indices]

    word_scores = sorted(word_scores, key=lambda x: abs(x[1]), reverse=True)

    return [w for w, s in word_scores[:top_n]]

# ---------- PREDICTION FUNCTION ----------
def predict_news(title, text):
    content = title + " " + text
    vec = vectorizer.transform([content])

    pred = model.predict(vec)[0]            # 0 or 1
    prob = model.predict_proba(vec).max() * 100

    label = "REAL" if pred == 1 else "FAKE"
    return label, round(prob, 2)

# ---------- API ----------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    title = data.get("title", "")
    text = data.get("text", "")

    if not title or not text:
        return jsonify({"error": "Title and text required"}), 400

    combined = title + " " + text

    # ----- Prediction -----
    vec = vectorizer.transform([combined])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]

    confidence = round(max(proba) * 100, 2)
    label = "REAL" if pred == 1 else "FAKE"

    # ----- Explainability (STEP 2B) -----
    top_words = get_top_words(combined, vectorizer, model)

    # ----- Risk Level -----
    if confidence >= 80:
        risk = "Low Risk"
    elif confidence >= 60:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    return jsonify({
        "label": label,
        "confidence": confidence,
        "risk": risk,
        "top_words": top_words
    })


if __name__ == "__main__":
    app.run(debug=True)
