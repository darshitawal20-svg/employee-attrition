from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import io
import os

app = Flask(__name__)

# Load Model and Features
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    FEATURE_COLUMNS = pickle.load(f)

# --- CONFIGURATION ---
MIN_WAGE_THRESHOLD = 500  # Any income below this is an automatic "Will Leave"

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

COLUMN_MAXS = {
    "Age": 60, "DailyRate": 1499, "DistanceFromHome": 29, "Education": 5,
    "EnvironmentSatisfaction": 4, "HourlyRate": 100, "JobInvolvement": 4,
    "JobLevel": 5, "JobSatisfaction": 4, "MonthlyIncome": 19999,
    "MonthlyRate": 26999, "NumCompaniesWorked": 9, "PercentSalaryHike": 25,
    "PerformanceRating": 4, "RelationshipSatisfaction": 4, "StockOptionLevel": 3,
    "TotalWorkingYears": 40, "TrainingTimesLastYear": 6, "WorkLifeBalance": 4,
    "YearsAtCompany": 40, "YearsInCurrentRole": 18, "YearsSinceLastPromotion": 15,
    "YearsWithCurrManager": 17,
}

# --- LOGIC FUNCTIONS ---

def common_sense_check(df):
    """
    Checks for unrealistic compensation. 
    If Income < $500, force 'Will Leave'.
    """
    income = pd.to_numeric(df.get("MonthlyIncome", 0), errors='coerce').fillna(0)
    # This returns a True/False mask for every row
    return (income < MIN_WAGE_THRESHOLD)

def normalize_values(df):
    df = df.copy()
    for col in COLUMN_MAXS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(COLUMN_DEFAULTS.get(col, 0))
            df[col] = df[col].clip(upper=COLUMN_MAXS[col])
    return df

def preprocess(df_raw):
    df = df_raw.copy()
    for col in ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours", "Attrition"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    df = normalize_values(df)

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0}).fillna(0)
    if "OverTime" in df.columns:
        df["OverTime"] = df["OverTime"].map({"Yes": 1, "No": 0}).fillna(0)

    cat_cols = ["BusinessTravel", "Department", "EducationField", "JobRole", "MaritalStatus"]
    for col in cat_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=[col], inplace=True)

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

    # Rule 1: Common sense check (e.g., is $10 < $500?)
    if common_sense_check(df_input).iloc[0]:
        return jsonify({
            "prediction": "Will Leave",
            "risk": "100.0%",
            "color": "red",
            "reason": "Compensation below realistic threshold."
        })

    # Rule 2: Normal AI Prediction
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
    df_raw = pd.read_csv(file) # Fast direct reading
    
    # Fast Auto-mapping
    mapping = {mc: uc for mc in MODEL_REQUIRED_COLUMNS for uc in df_raw.columns 
               if uc.strip().lower() == mc.lower()}

    df_mapped = pd.DataFrame()
    for mc in MODEL_REQUIRED_COLUMNS:
        if mapping.get(mc):
            df_mapped[mc] = df_raw[mapping[mc]].values
        else:
            df_mapped[mc] = COLUMN_DEFAULTS.get(mc, 0)

    # 1. Generate AI Predictions for all rows at once
    df_processed = preprocess(df_mapped)
    predictions = model.predict(df_processed)
    probabilities = model.predict_proba(df_processed)[:, 1]

    # 2. Vectorized Common Sense Override (FAST - No loops!)
    override_mask = (df_mapped["MonthlyIncome"] < MIN_WAGE_THRESHOLD).values
    predictions[override_mask] = 1
    probabilities[override_mask] = 1.0

    # 3. Format Results
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
    app.run(debug=True, host='0.0.0.0', port=5000)
