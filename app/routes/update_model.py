from flask import Blueprint, request, jsonify
from app.services import model_handler

bp = Blueprint('update_model', __name__, url_prefix='/update-model')

@bp.route('/', methods=['POST'])
def update_model():
    try:
        if 'model' not in request.files:
            return jsonify({"error": "Modelo no encontrado."}), 400
        if 'preprocessor' not in request.files:
            return jsonify({"error": "Preprocesador no encontrado."}), 400
        model_handler.update_model(request.files['model'], request.files['preprocessor'])
        return jsonify({"message": "Modelo actualizado."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
