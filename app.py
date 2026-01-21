import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import joblib
import numpy as np
import pandas as pd

# ======================
# App setup
# ======================
app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IS_VERCEL = os.getenv("VERCEL") == "1"

# ======================
# Lazy-loaded model
# ======================
model = None
feature_cols = None

def load_model():
    global model, feature_cols
    if model is None or feature_cols is None:
        model = joblib.load(os.path.join(BASE_DIR, "habitability_model.pkl"))
        feature_cols = joblib.load(os.path.join(BASE_DIR, "model_features.pkl"))
        print("✅ Model loaded")
        print("📌 Model features:", feature_cols)

# ======================
# Database helpers
# ======================
DB_NAME = os.path.join(BASE_DIR, "exoplanets.db")

def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pl_rade REAL,
            pl_bmasse REAL,
            pl_eqt REAL,
            pl_density REAL,
            pl_orbper REAL,
            pl_orbsmax REAL,
            st_luminosity REAL,
            pl_insol REAL,
            st_teff REAL,
            st_mass REAL,
            st_rad REAL,
            st_met REAL,
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
    return app.send_static_file("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        load_model()
        data = request.get_json(silent=True)

        if not data:
            return jsonify({"error": "No input data"}), 400

        # Normalize keys
        normalized = {k.strip().lower(): v for k, v in data.items()}

        # ✅ BUILD DATAFRAME (CRITICAL FIX)
        row = {}
        for col in feature_cols:
            key = col.lower()
            try:
                row[col] = float(normalized.get(key, 0.0))
            except Exception:
                row[col] = 0.0

        X = pd.DataFrame([row], columns=feature_cols)

        # Predict
        if hasattr(model, "predict_proba"):
            score = float(model.predict_proba(X)[0][1])
        else:
            score = float(model.predict(X)[0])

        return jsonify({
            "label": "Habitable" if score >= 0.7 else "Not Habitable",
            "score": round(score, 4)
        }), 200

    except Exception as e:
        print("🔥 PREDICT ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/ranking", methods=["GET"])
def ranking():
    try:
        if IS_VERCEL:
            return jsonify([]), 200

        conn = get_db()
        df = pd.read_sql("""
            SELECT
                pl_rade, pl_bmasse, pl_eqt, pl_density,
                pl_orbper, pl_orbsmax, st_luminosity,
                pl_insol, st_teff, st_mass, st_rad, st_met,
                score
            FROM predictions
            ORDER BY score DESC
            LIMIT 10
        """, conn)
        conn.close()

        if df.empty:
            return jsonify([]), 200

        return jsonify(df.to_dict(orient="records")), 200

    except Exception as e:
        print("🔥 RANKING ERROR:", e)
        return jsonify([]), 200

# ======================
# Local run only
# ======================
if __name__ == "__main__":
    app.run(debug=True)
