from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import io
import os

app = Flask(__name__)

# Load Model and Features
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        FEATURE_COLUMNS = pickle.load(f)
except Exception as e:
    print(f"Error loading model files: {e}")

# --- CONFIGURATION: COMMON SENSE THRESHOLDS ---
# If ANY of these fields are below these floors, we trigger "Will Leave"
MIN_MONTHLY_INCOME = 500
MIN_DAILY_RATE = 100
MIN_HOURLY_RATE = 10

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

COLUMN_DEFAULTS = {
    "Age": 37, "DailyRate": 800, "DistanceFromHome": 9, "Education": 3,
    "EnvironmentSatisfaction": 3, "HourlyRate": 65, "JobInvolvement": 3,
    "JobLevel": 2, "JobSatisfaction": 3, "MonthlyIncome": 6500,
    "MonthlyRate": 14000, "NumCompaniesWorked": 2, "PercentSalaryHike": 15,
    "PerformanceRating": 3, "RelationshipSatisfaction": 3, "StockOptionLevel": 1,
    "TotalWorkingYears": 11, "TrainingTimesLastYear": 3, "WorkLifeBalance": 3,
    "YearsAtCompany": 7, "YearsInCurrentRole": 4, "YearsSinceLastPromotion": 2,
    "YearsWithCurrManager": 4,
}

# --- LOGIC FUNCTIONS ---

def get_common_sense_mask(df):
    """
    Returns a True/False mask for rows that fail the common sense check.
    Checks Monthly Income, Daily Rate, and Hourly Rate.
    """
    income = pd.to_numeric(df.get("MonthlyIncome", 0), errors='coerce').fillna(0)
    daily = pd.to_numeric(df.get("DailyRate", 0), errors='coerce').fillna(0)
    hourly = pd.to_numeric(df.get("HourlyRate", 0), errors='coerce').fillna(0)
    
    # Logic: If Income < 500 OR Daily < 100 OR Hourly < 10 -> Force Leave
    return (income < MIN_MONTHLY_INCOME) | (daily < MIN_DAILY_RATE) | (hourly < MIN_HOURLY_RATE)

def preprocess(df_raw):
    df = df_raw.copy()
    
    # Clean up standard dropped columns
    for col in ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours", "Attrition"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Encode Gender and OverTime
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0}).fillna(0)
    if "OverTime" in df.columns:
        df["OverTime"] = df["OverTime"].map({"Yes": 1, "No": 0}).fillna(0)

    # One-Hot Encoding for categories
    cat_cols = ["BusinessTravel", "Department", "EducationField", "JobRole", "MaritalStatus"]
    for col in cat_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=[col], inplace=True)

    # Align to model's expected features
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return df

# --- ROUTES ---

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict_single", methods=["POST"])
def predict_single():
    data = request.json
    df_input = pd.DataFrame([data])

    # 1. Apply Common Sense Guardrail
    if get_common_sense_mask(df_input).iloc[0]:
        return jsonify({
            "prediction": "Will Leave",
            "risk": "98.5%",
            "color": "red",
            "reason": "Critical: Compensation rates (Daily/Hourly/Monthly) are unrealistically low."
        })

    # 2. Run ML Model
    df_processed = preprocess(df_input)
    prediction = model.predict(df_processed)[0]
    probability = model.predict_proba(df_processed)[0][1]

    return jsonify({
        "prediction": "Will Leave" if prediction == 1 else "Will Stay",
        "risk": f"{probability * 100:.1f}%",
        "color": "red" if prediction == 1 else "green"
    })

@app.route("/upload_and_predict", methods=["POST"])
def upload_and_predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files["file"]
    df_raw = pd.read_csv(file) # Direct stream read for speed
    
    # Auto-map uploaded columns to model requirements
    mapping = {mc: uc for mc in MODEL_REQUIRED_COLUMNS for uc in df_raw.columns 
               if uc.strip().lower() == mc.lower()}

    df_mapped = pd.DataFrame()
    for mc in MODEL_REQUIRED_COLUMNS:
        if mapping.get(mc):
            df_mapped[mc] = df_raw[mapping[mc]].values
        else:
            df_mapped[mc] = COLUMN_DEFAULTS.get(mc, 0)

    # 1. AI Predictions (Batch)
    df_processed = preprocess(df_mapped)
    predictions = model.predict(df_processed)
    probabilities = model.predict_proba(df_processed)[:, 1]

    # 2. Vectorized Common Sense Override (FAST)
    # Replaces slow loops with a single mask operation
    override_mask = get_common_sense_mask(df_mapped).values
    predictions[override_mask] = 1
    probabilities[override_mask] = 0.98

    # 3. Finalize Results
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
    # Debug mode allows auto-restart on save
    app.run(debug=True, host='0.0.0.0', port=5000)
