import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from supabase import create_client

# ======================
# Load env
# ======================
load_dotenv()

# ======================
# App setup
# ======================
app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IS_VERCEL = os.getenv("VERCEL") == "1"

# ======================
# Supabase (optional)
# ======================
supabase = None
if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY"):
    supabase = create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_KEY")
    )
    print("✅ Supabase connected")
else:
    print("⚠️ Supabase not configured")

# ======================
# Model globals
# ======================
model = None
feature_cols = None

# ======================
# SAFE model loader (NO dummy model)
# ======================
def load_model():
    global model, feature_cols

    if model is not None:
        return

    model_path = os.path.join(BASE_DIR, "habitability_model.pkl")
    features_path = os.path.join(BASE_DIR, "model_features.pkl")

    if not os.path.exists(model_path) or not os.path.exists(features_path):
        raise RuntimeError("❌ Trained model files missing in deployment")

    model = joblib.load(model_path)
    feature_cols = joblib.load(features_path)

    print("✅ Model loaded:", type(model).__name__)
    print("📋 Features:", feature_cols)

# ======================
# Routes
# ======================
@app.route("/")
def home():
    return send_from_directory("static", "index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    load_model()

    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data"}), 400

    normalized = {k.lower(): v for k, v in data.items()}

    values = []
    missing = []

    for col in feature_cols:
        if col not in normalized:
            missing.append(col)
            values.append(0.0)
        else:
            try:
                values.append(float(normalized[col]))
            except:
                missing.append(col)
                values.append(0.0)

    # ❌ DO NOT fake predictions
    if missing:
        return jsonify({
            "error": "Missing required features",
            "missing_features": missing
        }), 400

    X = pd.DataFrame([values], columns=feature_cols)

    # ✅ REAL prediction
    proba = model.predict_proba(X)
    score = float(proba[0][1])

    label = "Habitable" if score >= 0.7 else "Not Habitable"

    # Save to DB (Render only)
    if supabase and not IS_VERCEL:
        supabase.table("predictions").insert({
            "pl_name": normalized.get("pl_name", "Unknown"),
            "prediction_type": "habitability",
            "prediction_value": label,
            "confidence_score": round(score, 4)
        }).execute()

    return jsonify({
        "label": label,
        "score": round(score, 4),
        "confidence": (
            "High" if score >= 0.7 or score <= 0.3 else "Medium"
        )
    }), 200

@app.route("/ranking", methods=["GET"])
def ranking():
    if not supabase:
        return jsonify({"rankings": []}), 200

    response = supabase.table("predictions") \
        .select("*") \
        .order("confidence_score", desc=True) \
        .limit(100) \
        .execute()

    return jsonify({
        "rankings": response.data or []
    }), 200

# ======================
# Run local
# ======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
