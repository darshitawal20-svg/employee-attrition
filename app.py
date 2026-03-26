# ============================================================
#  STEP 2: Flask Web App — Employee Attrition Predictor
#  Run: python app.py  →  open http://localhost:5000
# ============================================================

from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import io
import joblib

app = Flask(__name__)

# --- Load the consolidated model package safely using joblib ---
try:
    data = joblib.load('attrition_model.pkl')
    model = data['model']
    FEATURE_COLUMNS = data['columns']
    print("✅ Model and Features loaded successfully!")
except FileNotFoundError:
    print("❌ Error: 'attrition_model.pkl' not found. Please run save_model.py first.")
except Exception as e:
    print(f"❌ An error occurred: {e}")
def preprocess(df_raw):
    """Apply same preprocessing as training."""
    df = df_raw.copy()

    # Drop irrelevant if present
    for col in ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Binary encoding
    if "Attrition" in df.columns:
        df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    if "OverTime" in df.columns:
        df["OverTime"] = df["OverTime"].map({"Yes": 1, "No": 0})

    # One-hot encoding
    cat_cols = ["BusinessTravel", "Department", "EducationField", "JobRole", "MaritalStatus"]
    for col in cat_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=[col], inplace=True)

    # Convert booleans
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    df[bool_cols] = df[bool_cols].astype(int)

    # Drop target if present
    if "Attrition" in df.columns:
        df.drop(columns=["Attrition"], inplace=True)

    # Align columns to training features (fill missing with 0)
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return df


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict_single", methods=["POST"])
def predict_single():
    data = request.json
    df_input = pd.DataFrame([data])
    df_processed = preprocess(df_input)
    prediction = model.predict(df_processed)[0]
    probability = model.predict_proba(df_processed)[0][1]
    return jsonify({
        "prediction": "Will Leave" if prediction == 1 else "Will Stay",
        "risk": f"{probability * 100:.1f}%",
        "color": "red" if prediction == 1 else "green"
    })


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    file = request.files["file"]
    df_raw = pd.read_csv(io.StringIO(file.stream.read().decode("utf-8")))
    df_processed = preprocess(df_raw)
    predictions = model.predict(df_processed)
    probabilities = model.predict_proba(df_processed)[:, 1]

    df_raw["Prediction"] = ["Will Leave" if p == 1 else "Will Stay" for p in predictions]
    df_raw["Attrition Risk %"] = (probabilities * 100).round(1)

    results = df_raw[["Prediction", "Attrition Risk %"]].to_dict(orient="records")
    return jsonify({"results": results, "total": len(results),
                    "leaving": int(sum(predictions)),
                    "staying": int(len(predictions) - sum(predictions))})


if __name__ == "__main__":
    print("🚀 Starting Employee Attrition Predictor...")
    print("   Open your browser at: http://localhost:5000")
    import os
app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
