import pandas as pd
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

class Preprocessor:
    def __init__(self, X):
        self.categorical_features = ['neighborhood']
        self.numerical_features = ['area_m2', 'bedrooms', 'bathrooms', 'parking', 'stratum', 'year_built']
        self.imputers = {}
        for column in self.numerical_features:
            self.imputers[column] = SimpleImputer(strategy='mean')
            self.imputers[column].fit(X[[column]])
        self.imputers['neighborhood'] = SimpleImputer(strategy='most_frequent')
        self.imputers['neighborhood'].fit(X[['neighborhood']])
        self.categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.categorical_encoder.fit(X[self.categorical_features])
        self.scaler = MinMaxScaler()
        self.scaler.fit(X[self.numerical_features])
        
    def transform(self, X):
        X = X.copy()
        for column in self.imputers:
            X[column] = self.imputers[column].transform(X[[column]])[:, 0]
        encoded_categorical = self.categorical_encoder.transform(X[self.categorical_features])
        encoded_categorical = pd.DataFrame(encoded_categorical, columns=self.categorical_encoder.get_feature_names_out(self.categorical_features), index=X.index)
        X = pd.concat([X, encoded_categorical], axis=1).drop(self.categorical_features, axis=1)
        X[self.numerical_features] = self.scaler.transform(X[self.numerical_features])
        return X
    
    def save(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)
 
    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)
