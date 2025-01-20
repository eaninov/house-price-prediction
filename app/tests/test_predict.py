import requests
import pytest

BASE_URL = "http://127.0.0.1:5000"

test_data_valid = {
    "area_m2": 90,
    "bedrooms": 3,
    "bathrooms": 2,
    "parking": 1,
    "stratum": 4,
    "year_built": 2000,
    "neighborhood": "Bosa"
}

test_data_invalid = {
    "area_m2": 90,
    "bedrooms": 3,
    "bathrooms": 2,
    "parking": 1,
    "stratum": 4,
    "year_built": 2000,
    # Falta 'neighborhood'
}

test_data_missing_field = {
    "area_m2": 90,
    "bedrooms": 3,
    "bathrooms": 2,
    "parking": 1,
    "stratum": 4,
    "year_built": 2000,
    "neighborhood": ""  # neighborhood vacío
}

def test_predict_valid_data():
    endpoint = f'{BASE_URL}/predict'
    response = requests.post(endpoint, json=test_data_valid)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data
    assert isinstance(data["predicted_price"], float)

def test_predict_invalid_data():
    endpoint = f'{BASE_URL}/predict'
    response = requests.post(endpoint, json=test_data_invalid)
    assert response.status_code == 400
    data = response.json()
    assert "error" in data

def test_predict_missing_field():
    endpoint = f'{BASE_URL}/predict'
    response = requests.post(endpoint, json=test_data_missing_field)
    assert response.status_code == 400
    data = response.json()
    assert "error" in data

if __name__ == "__main__":
    pytest.main()
