from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import io
import os

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    FEATURE_COLUMNS = pickle.load(f)

# The "Common Sense" Threshold
MIN_WAGE_THRESHOLD = 500 

MODEL_REQUIRED_COLUMNS = [
    "Age", "BusinessTravel", "DailyRate", "Department", "DistanceFromHome",
    "Education", "EducationField", "EnvironmentSatisfaction", "Gender",
    "HourlyRate", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction",
    "MaritalStatus", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
    "OverTime", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"
]

COLUMN_DEFAULTS = {col: 0 for col in MODEL_REQUIRED_COLUMNS}

def preprocess(df_raw):
    df = df_raw.copy()
    # Binary encoding
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0}).fillna(0)
    if "OverTime" in df.columns:
        df["OverTime"] = df["OverTime"].map({"Yes": 1, "No": 0}).fillna(0)

    # One-hot encoding
    cat_cols = ["BusinessTravel", "Department", "EducationField", "JobRole", "MaritalStatus"]
    for col in cat_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=[col], inplace=True)

    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return df

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict_single", methods=["POST"])
def predict_single():
    data = request.json
    income = float(data.get("MonthlyIncome", 0))

    # Common sense guardrail
    if income < MIN_WAGE_THRESHOLD:
        return jsonify({
            "prediction": "Will Leave",
            "risk": "98.0%",
            "color": "red",
            "reason": "Income below realistic threshold ($500)"
        })

    df_processed = preprocess(pd.DataFrame([data]))
    prediction = model.predict(df_processed)[0]
    probability = model.predict_proba(df_processed)[0][1]
    
    return jsonify({
        "prediction": "Will Leave" if prediction == 1 else "Will Stay",
        "risk": f"{probability * 100:.1f}%",
        "color": "red" if prediction == 1 else "green"
    })

@app.route("/upload_and_predict", methods=["POST"])
def upload_and_predict():
    file = request.files["file"]
    df_raw = pd.read_csv(file)
    
    # Auto-mapping
    mapping = {mc: uc for mc in MODEL_REQUIRED_COLUMNS for uc in df_raw.columns if uc.strip().lower() == mc.lower()}
    df_mapped = pd.DataFrame({mc: df_raw[mapping[mc]].values if mapping.get(mc) else COLUMN_DEFAULTS[mc] for mc in MODEL_REQUIRED_COLUMNS})

    # AI Prediction
    df_processed = preprocess(df_mapped)
    predictions = model.predict(df_processed)
    probabilities = model.predict_proba(df_processed)[:, 1]

    # Vectorized Guardrail for speed
    mask = (df_mapped["MonthlyIncome"] < MIN_WAGE_THRESHOLD).values
    predictions[mask] = 1
    probabilities[mask] = 0.98

    preview_cols = list(df_raw.columns[:3])
    df_result = df_raw[preview_cols].copy()
    df_result["Prediction"] = ["Will Leave" if p == 1 else "Will Stay" for p in predictions]
    df_result["Attrition Risk %"] = (probabilities * 100).round(1)

    return jsonify({
        "results": df_result.to_dict(orient="records"),
        "columns": preview_cols + ["Prediction", "Attrition Risk %"],
        "total": len(predictions),
        "leaving": int(sum(predictions)),
        "staying": int(len(predictions) - sum(predictions))
    })

if __name__ == "__main__":
    app.run(debug=True)
