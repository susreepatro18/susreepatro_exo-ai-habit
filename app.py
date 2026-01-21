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
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

DB_NAME = os.path.join(BASE_DIR, "exoplanets.db")

# ======================
# Database helpers (serverless-safe)
# ======================
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
    return "Backend is running successfully!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        load_model()   # ✅ THIS IS WHERE IT GOES

        data = request.json
        X = np.array([[float(data[col]) for col in feature_cols]])
        score = float(model.predict_proba(X)[0][1])

        conn = get_db()
        conn.execute("""
            INSERT INTO predictions (
                pl_rade, pl_bmasse, pl_eqt, pl_density,
                pl_orbper, pl_orbsmax, st_luminosity,
                pl_insol, st_teff, st_mass, st_rad, st_met, score
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            data["pl_rade"],
            data["pl_bmasse"],
            data["pl_eqt"],
            data["pl_density"],
            data["pl_orbper"],
            data["pl_orbsmax"],
            data["st_luminosity"],
            data["pl_insol"],
            data["st_teff"],
            data["st_mass"],
            data["st_rad"],
            data["st_met"],
            score
        ))
        conn.commit()
        conn.close()

        return jsonify({
            "label": "Habitable" if score >= 0.7 else "Not Habitable",
            "score": score
        })

    except Exception as e:
        print("PREDICT ERROR:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/ranking", methods=["GET"])
def ranking():
    try:
        conn = get_db()
        df = pd.read_sql("""
            SELECT
                pl_rade, pl_bmasse, pl_eqt, pl_density,
                pl_orbper, pl_orbsmax, st_luminosity,
                pl_insol, st_teff, st_mass, st_rad, st_met,
                score
            FROM predictions
            ORDER BY score DESC
        """, conn)
        conn.close()

        if df.empty:
            return jsonify([])

        df_unique = df.drop_duplicates(
            subset=[
                'pl_rade', 'pl_bmasse', 'pl_eqt', 'pl_density',
                'pl_orbper', 'pl_orbsmax', 'st_luminosity',
                'pl_insol', 'st_teff', 'st_mass', 'st_rad', 'st_met'
            ],
            keep='first'
        ).head(10)

        return jsonify(df_unique.to_dict(orient="records"))

    except Exception as e:
        print("RANKING ERROR:", e)
        return jsonify({"error": str(e)}), 500

# ======================
# Run server (local only)
# ======================
if __name__ == "__main__":
    app.run(debug=True)
