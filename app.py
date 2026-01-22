import os
import joblib
import numpy as np
import pandas as pd
import warnings
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()  # Load environment variables from .env file

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ======================
# App setup
# ======================
app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IS_VERCEL = os.getenv("VERCEL") == "1"

# Initialize Supabase client
url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_KEY')

if url and key:
    supabase: Client = create_client(url, key)
    print("✅ Supabase connected")
else:
    supabase = None
    print("⚠️ Supabase credentials not found")

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

    try:
        if not os.path.exists(model_path) or not os.path.exists(features_path):
            print("⚠️ Model files missing, creating dummy model...")
            create_dummy_model()

        model = joblib.load(model_path)
        feature_cols = joblib.load(features_path)

        print("✅ Model loaded successfully")
        print("🔍 Model class:", type(model).__name__)
        print("📌 Feature count:", len(feature_cols))
        print("📋 Features:", feature_cols)

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
            print("⚠️ Unknown model type, attempting to use anyway")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Creating fallback dummy model...")
        create_dummy_model()
        model = joblib.load(model_path)
        feature_cols = joblib.load(features_path)

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

    try:
        if not os.path.exists(model_path) or not os.path.exists(features_path):
            print("⚠️ Model files missing, creating dummy model...")
            create_dummy_model()

        model = joblib.load(model_path)
        feature_cols = joblib.load(features_path)

        print("✅ Model loaded successfully")
        print("🔍 Model class:", type(model).__name__)
        print("📌 Feature count:", len(feature_cols))
        print("📋 Features:", feature_cols)

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
            print("⚠️ Unknown model type, attempting to use anyway")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Creating fallback dummy model...")
        create_dummy_model()
        model = joblib.load(model_path)
        feature_cols = joblib.load(features_path)


# ======================
# Routes
# ======================
@app.route("/")
def home():
    return send_from_directory("static", "index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "supabase": "connected" if supabase else "disconnected",
        "models": "loaded" if model is not None else "not loaded"
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        load_model()

        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        print(f"📥 Received data: {data}")

        # Normalize keys to lowercase
        normalized = {k.lower().strip(): v for k, v in data.items()}

        # Extract values for model features
        values = []
        missing_features = []
        for col in feature_cols:
            try:
                val = normalized.get(col, None)
                if val is None:
                    val = 0.0
                    missing_features.append(col)
                else:
                    val = float(val)
                    if np.isnan(val) or np.isinf(val):
                        val = 0.0
                values.append(val)
            except (ValueError, TypeError):
                values.append(0.0)
                missing_features.append(col)

        if missing_features:
            print(f"⚠️ Missing features filled with 0: {missing_features}")

        X = pd.DataFrame([values], columns=feature_cols)
        print(f"✅ Input shape: {X.shape}")

        # Prediction with proper handling
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                score = float(proba[0][1]) if proba.shape[1] > 1 else float(proba[0][0])
            else:
                pred = model.predict(X)[0]
                score = float(pred) if isinstance(pred, (int, float, np.number)) else 0.5
        except Exception as pred_error:
            print(f"❌ Prediction failed: {pred_error}")
            score = 0.5

        # Save prediction to Supabase
        db_saved = False
        try:
            if supabase and not IS_VERCEL:
                planet_name = normalized.get('pl_name', f'Unknown-{score}')
                insert_data = {
                    "pl_name": planet_name,
                    "prediction_type": "habitability",
                    "prediction_value": "Habitable" if score >= 0.7 else "Not Habitable",
                    "confidence_score": round(score, 4)
                }
                print(f"💾 Saving to Supabase: {insert_data}")
                response = supabase.table('predictions').insert(insert_data).execute()
                print(f"✅ Prediction saved to Supabase: {response}")
                db_saved = True
            else:
                print(f"⚠️ Skipping Supabase save (supabase={supabase}, IS_VERCEL={IS_VERCEL})")
        except Exception as db_error:
            print(f"⚠️ Could not save to Supabase: {type(db_error).__name__}: {db_error}")
            import traceback
            traceback.print_exc()

        return jsonify({
            "label": "Habitable" if score >= 0.7 else "Not Habitable",
            "score": round(score, 4),
            "confidence": "High" if score >= 0.7 or score <= 0.3 else "Medium",
            "saved_to_db": db_saved,
            "planet_name": normalized.get('pl_name', 'Unknown')
        }), 200

    except Exception as e:
        print(f"🔥 PREDICT ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "type": type(e).__name__}), 500

@app.route("/ranking", methods=["GET"])
def ranking():
    try:
        print("📊 Ranking endpoint called")
        
        # Return empty list if Supabase not connected (will use fallback mock data in UI)
        if not supabase:
            print("⚠️ Supabase not connected")
            return jsonify({
                "message": "Supabase not connected",
                "total": 0,
                "rankings": []
            }), 200

        if IS_VERCEL:
            print("ℹ️ Running on Vercel")
            return jsonify({
                "message": "Mock data (Vercel environment)",
                "total": 0,
                "rankings": []
            }), 200

        # Try to get predictions from Supabase, ordered by confidence_score (highest first)
        try:
            print("🔄 Fetching from Supabase predictions table...")
            response = supabase.table('predictions').select('*').order('confidence_score', desc=True).limit(100).execute()
            rankings = response.data if response.data else []
            print(f"✅ Retrieved {len(rankings)} rankings from Supabase")
            
            return jsonify({
                "message": "Success" if rankings else "No predictions yet",
                "total": len(rankings),
                "rankings": rankings
            }), 200
            
        except Exception as table_error:
            print(f"⚠️ Error accessing predictions table: {table_error}")
            # If table doesn't exist yet, return empty
            return jsonify({
                "message": "Predictions table not ready yet - submit predictions first",
                "total": 0,
                "rankings": [],
                "error": str(table_error)
            }), 200
        
    except Exception as e:
        print(f"❌ RANKING ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "message": "Error fetching rankings",
            "total": 0,
            "rankings": [],
            "error": str(e)
        }), 200

# ======================
# Local run
# ======================
if __name__ == "__main__":
    print("🚀 Starting Flask app...")
    print(f"📁 Base directory: {BASE_DIR}")
    print(f"🔧 Models path: {os.path.join(BASE_DIR, 'habitability_model.pkl')}")
    print(f"🔧 Features path: {os.path.join(BASE_DIR, 'model_features.pkl')}")
    app.run(debug=True, host="0.0.0.0", port=5000)
