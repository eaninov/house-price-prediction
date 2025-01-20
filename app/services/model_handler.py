import pickle, os
import pandas as pd
from io import BytesIO
from app.services.preprocessor import Preprocessor
from config import Config
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score

model = None
preprocessor = None

def load_model(filepath):
    global model
    with open(filepath, "rb") as file:
        model = pickle.load(file)

def load_preprocessor(file_path):
    global preprocessor
    preprocessor = Preprocessor.load(file_path)

def predict_price(data):
    if model is None:
        raise ValueError("Modelo no cargado.")
    if preprocessor is None:
        raise ValueError("Preprocesador no cargado.")
    preprocessed_data = preprocessor.transform(pd.DataFrame([data]))
    return model.predict(preprocessed_data)[0]

def is_model_loaded():
    return model is not None

def is_preprocessor_loaded():
    return preprocessor is not None

def get_test_data():
    data = pd.read_csv(Config.DATA_PATH, encoding='cp1252')
    _, X, _, y = train_test_split(data.drop(['id', 'price'], axis=1), data['price'], 
                                  test_size=Config.TEST_SIZE, random_state=Config.SPLITTING_RANDOM_STATE)
    return X, y

def get_metrics():
    if model is None or preprocessor is None:
        raise ValueError("Modelo no cargado.")
    X, y = get_test_data()
    preprocessed_X = preprocessor.transform(X)
    y_pred = model.predict(preprocessed_X)
    return { 'MSE': mean_squared_error(y, y_pred),
             'RMSE': root_mean_squared_error(y, y_pred),
             'MAE': mean_absolute_error(y, y_pred),
             'R^2': r2_score(y, y_pred) }

def update_model(model_file, preprocessor_file):
    global model, preprocessor
    try:
        model = pickle.load(BytesIO(model_file.read()))
        preprocessor = pickle.load(BytesIO(preprocessor_file.read()))
    except Exception as e:
        raise ValueError(f"Error al actualizar el modelo: {e}")
