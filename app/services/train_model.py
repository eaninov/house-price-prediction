import pickle
from app.services.preprocessor import Preprocessor

def train_and_save_model(X, y, algorithm, model_file_path, preprocessor=None, preprocessor_file_path=None):
    if preprocessor is None: preprocessor = Preprocessor(X)
    preprocessed_X = preprocessor.transform(X)
    model = algorithm.fit(preprocessed_X, y)
    with open(model_file_path, 'wb') as file:
            pickle.dump(model, file)
    if preprocessor_file_path is not None:
        with open(preprocessor_file_path, 'wb') as file:
            pickle.dump(preprocessor, file)
    return model, preprocessor
    
