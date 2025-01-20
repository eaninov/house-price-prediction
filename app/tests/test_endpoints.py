import requests
import json

BASE_URL = 'http://127.0.0.1:5000'

test_data = [{
    'area_m2': 120,
    'bedrooms': 3,
    'bathrooms': 2,
    'parking': 1,
    'stratum': 4,
    'year_built': 2005,
    'neighborhood': 'Bosa'
}, 
{
    'area_m2': 78,
    'bedrooms': 1,
    'bathrooms': 1,
    'parking': 0,
    'stratum': 1,
    'year_built': 1980,
    'neighborhood': 'Suba'
}]

def test_predict():
    print('/predict')
    endpoint = f'{BASE_URL}/predict'
    response = requests.post(endpoint, json=test_data[0])
    if response.status_code == 200:
        print('Response:', response.json())
    else:
        print(f'Status code: {response.status_code}, {response.text}')

    response = requests.post(endpoint, json=test_data[1])
    if response.status_code == 200:
        print('Response:', response.json())
    else:
        print(f'Status code: {response.status_code}, {response.text}')

def test_metrics():
    print('/metrics')
    endpoint = f'{BASE_URL}/metrics'
    response = requests.get(endpoint)
    if response.status_code == 200:
        print('Response:', response.json())
    else:
        print(f'Status code: {response.status_code}, {response.text}')

def test_health():
    print('/health')
    endpoint = f'{BASE_URL}/health'
    response = requests.get(endpoint)
    if response.status_code == 200:
        print('Response:', response.json())
    else:
        print(f'Status code: {response.status_code}, {response.text}')

def test_update_model():
    print('/update-model')
    endpoint = f'{BASE_URL}/update-model'
    model_file_path = 'models/xgboost.pkl'
    preprocessor_file_path = 'models/preprocessor.pkl'
    with open(model_file_path, 'rb') as model_file:
        with open(preprocessor_file_path, 'rb') as preprocessor_file:
            response = requests.post(endpoint, files={'model': model_file, 'preprocessor': preprocessor_file})
    if response.status_code == 200:
        print('Response:', response.json())
    else:
        print(f'Status code: {response.status_code}, {response.text}')

if __name__ == '__main__':
    test_health()    
    test_metrics()  
    test_predict()   
    test_update_model()
    test_metrics()
