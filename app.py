from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)
import io

app = Flask(__name__)
CORS(app)  # Permite peticiones desde GitHub Pages

# Cargar modelos al iniciar
with open("logistic_model.pkl", "rb") as f:
    logreg = pickle.load(f)

with open("mlp_model.pkl", "rb") as f:
    mlp = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Columnas del dataset (sin ID y sin la variable objetivo)
FEATURE_COLUMNS = [
    "ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]

def get_model(model_name):
    if model_name == "logistic":
        return logreg
    elif model_name == "mlp":
        return mlp
    else:
        return None

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "API funcionando correctamente"})

@app.route("/predict/individual", methods=["POST"])
def predict_individual():
    try:
        data = request.get_json()
        model_name = data.get("model", "logistic")
        features = data.get("features", {})

        model = get_model(model_name)
        if model is None:
            return jsonify({"error": "Modelo no válido"}), 400

        # Construir vector de features en orden correcto
        input_data = np.array([[features[col] for col in FEATURE_COLUMNS]])
        input_scaled = scaler.transform(input_data)

        prediction = int(model.predict(input_scaled)[0])
        probabilities = model.predict_proba(input_scaled)[0].tolist()

        return jsonify({
            "prediction": prediction,
            "label": "Default" if prediction == 1 else "No Default",
            "probability_no_default": round(probabilities[0] * 100, 2),
            "probability_default": round(probabilities[1] * 100, 2),
            "model": model_name
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    try:
        model_name = request.form.get("model", "logistic")
        file = request.files.get("file")

        if not file:
            return jsonify({"error": "No se envió ningún archivo"}), 400

        model = get_model(model_name)
        if model is None:
            return jsonify({"error": "Modelo no válido"}), 400

        # Leer CSV
        df = pd.read_csv(io.StringIO(file.read().decode("utf-8")))

        # Verificar si tiene columna objetivo
        has_labels = "default payment next month" in df.columns

        # Obtener features
        X = df[FEATURE_COLUMNS].values
        X_scaled = scaler.transform(X)

        predictions = model.predict(X_scaled).tolist()

        result = {"predictions": predictions, "model": model_name, "total": len(predictions)}

        if has_labels:
            y_true = df["default payment next month"].values
            cm = confusion_matrix(y_true, predictions).tolist()
            result["confusion_matrix"] = cm
            result["metrics"] = {
                "accuracy": round(accuracy_score(y_true, predictions) * 100, 2),
                "precision": round(precision_score(y_true, predictions, zero_division=0) * 100, 2),
                "recall": round(recall_score(y_true, predictions, zero_division=0) * 100, 2),
                "f1_score": round(f1_score(y_true, predictions, zero_division=0) * 100, 2),
            }
            result["has_labels"] = True
        else:
            result["has_labels"] = False

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False)
