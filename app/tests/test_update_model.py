import requests
import pytest
import pickle
from app.services.model_handler import model, preprocessor 

BASE_URL = "http://127.0.0.1:5000"

def test_update_model_1():
    print(model)
    endpoint = f'{BASE_URL}/update-model'
    files = {
        'model': open('models/gradient_boosting.pkl', 'rb'),
        'preprocessor': open('models/preprocessor.pkl', 'rb')
    }
    response = requests.post(endpoint, files=files)
    assert response.status_code == 200
    data = response.json()
    assert 'message' in data
    assert data['message'] == "Modelo actualizado."
    files['model'].close()
    files['preprocessor'].close()

# Ejecutar las pruebas
if __name__ == "__main__":
    pytest.main()
