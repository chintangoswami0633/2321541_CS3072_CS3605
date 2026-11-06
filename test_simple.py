#!/usr/bin/env python3
import requests
import json

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

print("Testing /api/test endpoint...")
try:
    response = requests.post(
        "http://localhost:8000/api/test",
        json=test_data
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")
