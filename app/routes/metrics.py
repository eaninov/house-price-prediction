from flask import Blueprint, jsonify
from app.services import model_handler

bp = Blueprint('metrics', __name__, url_prefix='/metrics')

@bp.route('/', methods=['GET'])
def metrics():
    try:
        test_metrics = model_handler.get_metrics()
        if test_metrics is None:
            return jsonify({"error": "No es posible evaluar las métricas del modelo."}), 500
        return jsonify(test_metrics), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
