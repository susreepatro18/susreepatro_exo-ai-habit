import os
import sqlite3
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ======================
# App setup
# ======================
app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IS_VERCEL = os.getenv("VERCEL") == "1"

# ======================
# Model globals
# ======================
model = None
feature_cols = None

# ======================
# Model utilities
# ======================
def is_valid_forest(m):
    """Ensure RandomForest is actually trained"""
    return (
        hasattr(m, "estimators_")
        and isinstance(m.estimators_, list)
        and len(m.estimators_) > 0
    )

def create_dummy_model():
    """Failsafe model if pkl is corrupted or missing"""
    from sklearn.ensemble import RandomForestClassifier

    print("⚠️ Creating fallback dummy model")

    features = [
        "pl_rade", "pl_bmasse", "pl_eqt", "pl_density",
        "pl_orbper", "pl_orbsmax", "st_luminosity",
        "pl_insol", "st_teff", "st_mass", "st_rad", "st_met"
    ]

    X = np.random.rand(200, len(features))
    y = np.random.randint(0, 2, 200)

    m = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42
    )
    m.fit(X, y)

    joblib.dump(m, os.path.join(BASE_DIR, "habitability_model.pkl"))
    joblib.dump(features, os.path.join(BASE_DIR, "model_features.pkl"))
    

    print("✅ Dummy model created successfully")

def load_model():
    global model, feature_cols

    if model is not None and feature_cols is not None:
        return

    model_path = os.path.join(BASE_DIR, "habitability_model.pkl")
    features_path = os.path.join(BASE_DIR, "model_features.pkl")

    if not (os.path.exists(model_path) and os.path.exists(features_path)):
        raise RuntimeError("❌ Model files missing")

    model = joblib.load(model_path)
    feature_cols = joblib.load(features_path)

    print("✅ Model loaded")
    print("🔍 Model class:", type(model))
    print("📌 Feature count:", len(feature_cols))

    # ✅ Model-specific validation
    if model.__class__.__name__ == "XGBClassifier":
        if not hasattr(model, "n_estimators") or model.n_estimators <= 0:
            raise RuntimeError("❌ Invalid XGBoost model")
        print(f"🚀 XGBoost model with {model.n_estimators} estimators ready")

    elif hasattr(model, "estimators_"):
        if len(model.estimators_) == 0:
            raise RuntimeError("❌ RandomForest has zero trees")
        print(f"🌲 RandomForest with {len(model.estimators_)} trees ready")

    else:
        raise RuntimeError("❌ Unsupported model type")


# ======================
# Database (local only)
# ======================
DB_NAME = os.path.join(BASE_DIR, "exoplanets.db")

def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            score REAL
        )
    """)
    conn.commit()
    return conn

# ======================
# Routes
# ======================
@app.route("/")
def home():
    return send_from_directory("static", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        load_model()

        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "No input data"}), 400

        normalized = {k.lower().strip(): v for k, v in data.items()}

        values = []
        for col in feature_cols:
            try:
                val = float(normalized.get(col, 0.0))
                if np.isnan(val) or np.isinf(val):
                    val = 0.0
                values.append(val)
            except Exception:
                values.append(0.0)

        X = pd.DataFrame([values], columns=feature_cols)

        # ✅ XGBoost-safe prediction
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            score = float(proba[0][1]) if proba.shape[1] > 1 else float(proba[0][0])
        else:
            score = float(model.predict(X)[0])

        return jsonify({
            "label": "Habitable" if score >= 0.7 else "Not Habitable",
            "score": round(score, 4)
        }), 200

    except Exception as e:
        print("🔥 PREDICT ERROR:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/ranking", methods=["GET"])
def ranking():
    if IS_VERCEL:
        return jsonify([]), 200

    try:
        conn = get_db()
        df = pd.read_sql(
            "SELECT score FROM predictions ORDER BY score DESC LIMIT 10",
            conn
        )
        conn.close()
        return jsonify(df.to_dict(orient="records")), 200
    except Exception:
        return jsonify([]), 200

# ======================
# Local run
# ======================
if __name__ == "__main__":
    app.run(debug=False)
