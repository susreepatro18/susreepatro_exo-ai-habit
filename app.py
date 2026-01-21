import os
from flask import Flask, request, jsonify, send_from_directory
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
        model_path = os.path.join(BASE_DIR, "habitability_model.pkl")
        features_path = os.path.join(BASE_DIR, "model_features.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(features_path):
            print("⚠️ Model files not found. Creating dummy model...")
            create_dummy_model()
        
        model = joblib.load(model_path)
        feature_cols = joblib.load(features_path)
        print("✅ Model loaded")
        print("📌 Model features:", feature_cols)

def create_dummy_model():
    """Create a minimal dummy model for deployment without trained files"""
    from sklearn.ensemble import RandomForestClassifier
    
    feature_cols = [
        "pl_rade", "pl_bmasse", "pl_eqt", "pl_density",
        "pl_orbper", "pl_orbsmax", "st_luminosity",
        "pl_insol", "st_teff", "st_mass", "st_rad", "st_met"
    ]
    
    X = np.random.rand(100, 12)
    y = np.random.randint(0, 2, 100)
    
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X, y)
    
    joblib.dump(model, os.path.join(BASE_DIR, "habitability_model.pkl"))
    joblib.dump(feature_cols, os.path.join(BASE_DIR, "model_features.pkl"))
    print("✅ Dummy model created")

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
    return send_from_directory("static", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        load_model()

        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "No input data"}), 400

        # Normalize keys
        normalized = {k.strip().lower(): v for k, v in data.items()}

        values = []
        for col in feature_cols:
            try:
                values.append(float(normalized.get(col.lower(), 0.0)))
            except ValueError:
                values.append(0.0)

        # ✅ CRITICAL FIX: DataFrame, not NumPy
        X = pd.DataFrame([values], columns=feature_cols)

        if hasattr(model, "predict_proba"):
            score = float(model.predict_proba(X)[0][1])
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
