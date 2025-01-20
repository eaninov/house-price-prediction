from flask import Blueprint, jsonify
from app.services import model_handler

bp = Blueprint('health', __name__, url_prefix='/health')

@bp.route('/', methods=['GET'])
def health_check():
    try:
        model_ok = model_handler.is_model_loaded()
        preprocessor_ok = model_handler.is_preprocessor_loaded()
        status = {
            "status": "healthy" if model_ok and preprocessor_ok else "unhealthy",
            "model_loaded": model_ok,
            "preprocessor_loaded": preprocessor_ok
        }
        return jsonify(status), 200 if model_ok and preprocessor_ok else 500
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500
