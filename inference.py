import os
import joblib
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "mlp-pipeline.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
MODEL = joblib.load(MODEL_PATH)
print("[INFO] Model loaded successfully from", MODEL_PATH)

app = Flask("Heart Failure MLP Optuna - Inference Server")

@app.route("/", methods=["GET"])
def root():
    return jsonify({"status": "Server is Running"})

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "Ok", "model_loaded": MODEL is not None})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        instances = data.get("instances")
        return_proba = data.get("return_proba", True)

        if instances is None:
            return jsonify({"error": "Missing 'Instances' Field"}), 400

        X = np.asarray(instances, dtype=np.float32)
        preds = MODEL.predict(X).tolist()
        resp = {"predictions": preds}

        if return_proba and hasattr(MODEL, "predict_proba"):
            resp["probabilities"] = MODEL.predict_proba(X).tolist()

        return jsonify(resp)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="localhost", port=port, debug=False)