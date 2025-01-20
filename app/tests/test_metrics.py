import requests
import pytest

BASE_URL = "http://127.0.0.1:5000"

def test_metrics():
    endpoint = f'{BASE_URL}/metrics'
    response = requests.get(endpoint)
    assert response.status_code == 200
    data = response.json()
    assert "RMSE" in data
    assert "MSE" in data
    assert "MAE" in data
    assert "R^2" in data
    assert isinstance(data["MAE"], (float, int))
    assert isinstance(data["MSE"], (float, int))
    assert isinstance(data["RMSE"], (float, int))
    assert isinstance(data["R^2"], (float, int))
    assert data["MAE"] >= 0
    assert data["MSE"] >= 0
    assert data["RMSE"] >= 0
    assert data["R^2"] <= 1
    
if __name__ == "__main__":
    pytest.main()
