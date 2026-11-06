#!/usr/bin/env python3
"""Test script to verify the /api/predict endpoint"""

import requests
import json

# Test data that should work
test_data = {
    "age": 21,
    "height_cm": 165,
    "weight_kg": 67,
    "gender": "Male",
    "chest_cm": 110,
    "biceps_cm": 40,
    "waist_cm": 100,
    "training_weeks": 48,
    "training_frequency": 7,
    "training_intensity": "High",
    "diet_controlled": True,
    "commitment": 5
}

print("Testing /api/predict endpoint...")
print(f"Sending data: {json.dumps(test_data, indent=2)}")

try:
    response = requests.post(
        "http://localhost:8000/api/predict",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        print("✅ SUCCESS!")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print("❌ ERROR!")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
except requests.exceptions.ConnectionError:
    print("❌ ERROR: Could not connect to backend server")
    print("Make sure the backend is running on http://localhost:8000")
except Exception as e:
    print(f"❌ ERROR: {e}")
