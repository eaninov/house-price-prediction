import requests
import pytest

BASE_URL = "http://127.0.0.1:5000"

def test_health():
    endpoint = f'{BASE_URL}/health'
    response = requests.get(endpoint)
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "preprocessor_loaded" in data
    assert data["status"] == "healthy"
    assert data["model_loaded"]
    assert data["preprocessor_loaded"]

# Ejecutar las pruebas
if __name__ == "__main__":
    pytest.main()
