import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# Supabase is optional – app still runs if SDK or env is missing
try:
    from supabase import create_client
except ImportError:
    create_client = None

# ======================
# Load environment
# ======================
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

IS_VERCEL = os.getenv("VERCEL") == "1"

# ======================
# Supabase (optional)
# ======================
supabase = None
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if create_client and SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase connected")
    except Exception as e:
        print("⚠️ Supabase init failed:", e)
        supabase = None
else:
    print("⚠️ Supabase not configured or supabase-py not installed")

# ======================
# Model globals
# ======================
model = None
feature_cols = None


def load_model():
    """
    Lazy-load model and feature list.
    Safe in serverless environments.
    """
    global model, feature_cols

    if model is not None:
        return

    model_path = os.path.join(BASE_DIR, "habitability_model.pkl")
    features_path = os.path.join(BASE_DIR, "model_features.pkl")

    if not os.path.exists(model_path):
        raise RuntimeError("habitability_model.pkl missing in project root")

    if not os.path.exists(features_path):
        raise RuntimeError("model_features.pkl missing in project root")

    loaded_model = joblib.load(model_path)
    loaded_features = joblib.load(features_path)

    model = loaded_model
    feature_cols = list(loaded_features)

    print("✅ Model loaded:", type(model).__name__)
    print("📋 Features:", feature_cols)
    print("🔍 Has predict_proba:", hasattr(model, "predict_proba"))


# ======================
# Routes
# ======================

@app.route("/")
def home():
    # Serve your front-end (static/index.html)
    return send_from_directory("static", "index.html")


@app.route("/health")
def health():
    ok = True
    msg = "ok"
    try:
        load_model()
    except Exception as e:
        ok = False
        msg = str(e)

    return jsonify({
        "status": "ok" if ok else "error",
        "message": msg,
        "model_loaded": ok
    }), 200 if ok else 500


@app.route("/predict", methods=["POST"])
def predict():
    # Ensure model is loaded
    try:
        load_model()
    except Exception as e:
        return jsonify({
            "error": "Model not available",
            "details": str(e)
        }), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data"}), 400

    # Normalize input keys to lowercase
    normalized = {str(k).lower(): v for k, v in data.items()}

    values = []
    missing = []

    # Expect the same feature set as used in training
    for col in feature_cols:
        key = col.lower()
        if key not in normalized:
            missing.append(col)
            values.append(0.0)
        else:
            try:
                values.append(float(normalized[key]))
            except Exception:
                missing.append(col)
                values.append(0.0)

    if missing:
        return jsonify({
            "error": "Missing or invalid required features",
            "missing_features": missing
        }), 400

    X = pd.DataFrame([values], columns=feature_cols)

    # Safe prediction
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            score = float(proba[0][1])
        else:
            pred = model.predict(X)[0]
            score = float(pred)
    except Exception as e:
        return jsonify({
            "error": "Model prediction failed",
            "details": str(e)
        }), 500

    label = "Habitable" if score >= 0.7 else "Not Habitable"
    confidence = "High" if score >= 0.7 or score <= 0.3 else "Medium"

    # Optional: log to Supabase (skip on Vercel if you prefer)
    if supabase and not IS_VERCEL:
        try:
            pl_name = normalized.get("pl_name", "Unknown")
            supabase.table("predictions").insert({
                "pl_name": pl_name,
                "prediction_type": "habitability",
                "prediction_value": label,
                "confidence_score": round(score, 4)
            }).execute()
        except Exception as e:
            print("⚠️ Supabase insert failed:", e)

    return jsonify({
        "label": label,
        "score": round(score, 4),
        "confidence": confidence
    }), 200


@app.route("/ranking", methods=["GET"])
def ranking():
    if not supabase:
        return jsonify({"rankings": []}), 200

    try:
        response = (
            supabase
            .table("predictions")
            .select("*")
            .order("confidence_score", desc=True)
            .limit(100)
            .execute()
        )
        return jsonify({"rankings": response.data or []}), 200
    except Exception as e:
        return jsonify({
            "rankings": [],
            "error": str(e)
        }), 200


# Local dev only – Vercel/Render will import `app`
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
