from flask import Flask
from app.services.preprocessor import Preprocessor
from app.services import model_handler
from app.routes import predict, metrics, health, update_model
from config import Config

def create_app():
    app = Flask(__name__)
    
    app.config.from_object(Config)

    try:
        model_handler.load_preprocessor(app.config["PREPROCESSOR_PATH"])
        model_handler.load_model(app.config["MODEL_PATH"])
    except Exception as e:
        app.logger.error(f"Error al cargar el modelo: {e}")
        raise

    app.register_blueprint(predict.bp)
    app.register_blueprint(metrics.bp)
    app.register_blueprint(health.bp)
    app.register_blueprint(update_model.bp)
    return app
