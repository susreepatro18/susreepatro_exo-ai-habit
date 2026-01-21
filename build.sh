#!/bin/bash
# Install dependencies
pip install -r requirements.txt

# If model files don't exist, create a default one
if [ ! -f "habitability_model.pkl" ]; then
    python -c "
import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Create a dummy model as fallback
model = RandomForestClassifier(n_estimators=10, random_state=42)
X = np.random.rand(10, 12)
y = np.random.randint(0, 2, 10)
model.fit(X, y)
joblib.dump(model, 'habitability_model.pkl')

features = ['pl_rade','pl_bmasse','pl_eqt','pl_density','pl_orbper','pl_orbsmax','st_luminosity','pl_insol','st_teff','st_mass','st_rad','st_met']
joblib.dump(features, 'model_features.pkl')
print('✅ Default models created')
"
fi
