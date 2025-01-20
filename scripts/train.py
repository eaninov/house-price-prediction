import pandas as pd
import os
from app.services.train_model import train_and_save_model
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from config import Config
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score

# Los hiperparámetros de estos algoritmos fueron ajustados usando grid search + cross validation.
# Sin embargo, se requiere mayor experimentación.
algorithms = {  'dummy': DummyRegressor(strategy='mean'),
                'ridge': Ridge(random_state=100, alpha=0.001),
                'lasso': Lasso(random_state=100, alpha=1000),
                'decision_tree': DecisionTreeRegressor(criterion='squared_error', splitter='best', random_state=100,
                                                       max_depth=None, min_impurity_decrease=0.0, min_samples_split=10),
                'random_forest': RandomForestRegressor(n_estimators=100, criterion='squared_error', max_features=None,
                                                       n_jobs=-1, random_state=100, max_depth=None,
                                                       min_impurity_decrease=0.0, min_samples_split=2),
                'mlp': MLPRegressor(max_iter=50000, learning_rate='adaptive', warm_start=True, early_stopping=True,
                                    random_state=100, activation='relu', alpha=0.0001, hidden_layer_sizes=(13, 13, 13, 13)),
                'knn': KNeighborsRegressor(n_neighbors=5, p=2, weights='distance'),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=5,
                                                               min_impurity_decrease=0.001, min_samples_split=10),   
                'xgboost': XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1, 
                                        max_depth=None, alpha=10, n_estimators=500)
              }

def get_training_data():
    data = pd.read_csv(Config.DATA_PATH, encoding='cp1252')
    X, _, y, _ = train_test_split(data.drop(['id', 'price'], axis=1), data['price'],
                                  test_size=Config.TEST_SIZE, random_state=Config.SPLITTING_RANDOM_STATE)
    return X, y

def create_model(algorithm='ridge'):
    X, y = get_training_data()
    model, preprocessor = train_and_save_model(X, y, algorithms[algorithm], os.path.join(Config.MODEL_DIR, f'{algorithm}.pkl'), 
                                               None, os.path.join(Config.MODEL_DIR, 'preprocessor.pkl'))
    print(f'Modelo {algorithm} entrenado.')
    preprocessed_X = preprocessor.transform(X)
    y_pred = model.predict(preprocessed_X)
    print ( {'MSE': mean_squared_error(y, y_pred),
             'RMSE': root_mean_squared_error(y, y_pred),
             'MAE': mean_absolute_error(y, y_pred),
             'R^2': r2_score(y, y_pred) } )

if __name__ == '__main__':
    create_model()