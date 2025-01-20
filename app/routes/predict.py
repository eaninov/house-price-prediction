from flask import Blueprint, request, jsonify
from app.services.model_handler import predict_price
from app.services.utils import validate_input

bp = Blueprint("predict", __name__, url_prefix="/predict")

@bp.route("/", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not validate_input(data):
            return jsonify({"error": "Datos inválidos"}), 400
        prediction = predict_price(data)
        return jsonify({"predicted_price": float(prediction)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
