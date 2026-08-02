from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.environ.get("MODEL_PATH", "superkart_best_model.joblib")
model = joblib.load(MODEL_PATH)

FEATURE_COLUMNS = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category"
]

PERISHABLES = [
    "Dairy",
    "Meat",
    "Fruits and Vegetables",
    "Breads",
    "Breakfast",
    "Seafood",
    "Frozen Foods"
]

def prepare_input(df):
    """Prepare input data for model prediction."""

    df = df.copy()

    # Create Product_Id_char from Product_Id if raw Product_Id is available
    if "Product_Id" in df.columns and "Product_Id_char" not in df.columns:
        df["Product_Id_char"] = df["Product_Id"].astype(str).str[:2]

    # Create Store_Age_Years from Store_Establishment_Year if raw year is available
    if "Store_Establishment_Year" in df.columns and "Store_Age_Years" not in df.columns:
        df["Store_Age_Years"] = 2025 - df["Store_Establishment_Year"].astype(int)

    # Create Product_Type_Category from Product_Type if raw Product_Type is available
    if "Product_Type" in df.columns and "Product_Type_Category" not in df.columns:
        df["Product_Type_Category"] = np.where(
            df["Product_Type"].isin(PERISHABLES),
            "Perishables",
            "Non Perishables"
        )

    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns after feature engineering: {missing_cols}")

    return df[FEATURE_COLUMNS]

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "SuperKart Sales Forecasting API is running.",
        "deployment": "GitHub Codespaces using Docker Compose",
        "endpoints": ["/health", "/predict", "/batch_predict"]
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_json = request.get_json()

        if input_json is None:
            return jsonify({"error": "No JSON payload received"}), 400

        input_df = pd.DataFrame([input_json])
        input_df = prepare_input(input_df)

        prediction = model.predict(input_df)[0]

        return jsonify({
            "prediction": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    try:
        input_json = request.get_json()

        if input_json is None:
            return jsonify({"error": "No JSON payload received"}), 400

        records = input_json.get("records", [])

        if len(records) == 0:
            return jsonify({"error": "No records found for batch prediction"}), 400

        input_df = pd.DataFrame(records)
        input_df_prepared = prepare_input(input_df)

        predictions = model.predict(input_df_prepared)

        output_df = input_df.copy()
        output_df["Predicted_Product_Store_Sales_Total"] = predictions.round(2)

        return jsonify({
            "predictions": output_df.to_dict(orient="records")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
