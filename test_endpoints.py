#!/usr/bin/env python3
"""Test script for Flask API endpoints"""

import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()

BASE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("🏥 Testing /health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"❌ Error: {e}")
    print()

def test_predict():
    """Test predict endpoint with sample data"""
    print("🔮 Testing /predict endpoint...")
    
    sample_data = {
        "pl_name": "Test-Exoplanet-1",
        "pl_rade": 1.5,
        "pl_bmasse": 2.0,
        "pl_eqt": 350,
        "pl_density": 5.5,
        "pl_orbper": 365.25,
        "pl_orbsmax": 1.0,
        "st_luminosity": 1.0,
        "pl_insol": 1.0,
        "st_teff": 5778,
        "st_mass": 1.0,
        "st_rad": 1.0,
        "st_met": 0.0
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=sample_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"❌ Error: {e}")
    print()

def test_ranking():
    """Test ranking endpoint"""
    print("📊 Testing /ranking endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/ranking")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"❌ Error: {e}")
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Flask API Endpoint Test Suite")
    print("=" * 60)
    print()
    
    # Make sure the server is running
    print("⚠️  Make sure Flask app is running: python app.py")
    print()
    
    input("Press Enter to start tests...")
    print()
    
    test_health()
    test_predict()
    test_ranking()
    
    print("=" * 60)
    print("✅ Tests completed!")
    print("=" * 60)
