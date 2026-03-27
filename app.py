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

MODEL_COLUMN_DESCRIPTIONS = {
    "Age": "Employee age (number)",
    "BusinessTravel": "Travel frequency (Travel_Rarely / Travel_Frequently / Non-Travel)",
    "DailyRate": "Daily rate / daily wage (number)",
    "Department": "Department name (Sales / Research & Development / Human Resources)",
    "DistanceFromHome": "Distance from home in km/miles (number)",
    "Education": "Education level 1-5 (number)",
    "EducationField": "Field of education (Life Sciences / Medical / Marketing etc.)",
    "EnvironmentSatisfaction": "Environment satisfaction 1-4 (number)",
    "Gender": "Gender (Male / Female)",
    "HourlyRate": "Hourly rate (number)",
    "JobInvolvement": "Job involvement 1-4 (number)",
    "JobLevel": "Job level 1-5 (number)",
    "JobRole": "Job role / designation",
    "JobSatisfaction": "Job satisfaction 1-4 (number)",
    "MaritalStatus": "Marital status (Single / Married / Divorced)",
    "MonthlyIncome": "Monthly salary/income (number)",
    "MonthlyRate": "Monthly rate (number)",
    "NumCompaniesWorked": "Number of previous companies (number)",
    "OverTime": "Works overtime? (Yes / No)",
    "PercentSalaryHike": "Salary hike percentage (number)",
    "PerformanceRating": "Performance rating 1-4 (number)",
    "RelationshipSatisfaction": "Relationship satisfaction 1-4 (number)",
    "StockOptionLevel": "Stock option level 0-3 (number)",
    "TotalWorkingYears": "Total years of work experience (number)",
    "TrainingTimesLastYear": "Training sessions last year (number)",
    "WorkLifeBalance": "Work life balance 1-4 (number)",
    "YearsAtCompany": "Years at current company (number)",
    "YearsInCurrentRole": "Years in current role (number)",
    "YearsSinceLastPromotion": "Years since last promotion (number)",
    "YearsWithCurrManager": "Years with current manager (number)"
}

# Safe default values — used when a column is missing from uploaded CSV
# These are average/median values from IBM HR dataset
COLUMN_DEFAULTS = {
    "Age": 37,
    "DailyRate": 800,
    "DistanceFromHome": 9,
    "Education": 3,
    "EnvironmentSatisfaction": 3,
    "HourlyRate": 65,
    "JobInvolvement": 3,
    "JobLevel": 2,
    "JobSatisfaction": 3,
    "MonthlyIncome": 6500,
    "MonthlyRate": 14000,
    "NumCompaniesWorked": 2,
    "PercentSalaryHike": 15,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 3,
    "StockOptionLevel": 1,
    "TotalWorkingYears": 11,
    "TrainingTimesLastYear": 3,
    "WorkLifeBalance": 3,
    "YearsAtCompany": 7,
    "YearsInCurrentRole": 4,
    "YearsSinceLastPromotion": 2,
    "YearsWithCurrManager": 4,
}

# Maximum realistic values from IBM HR dataset
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

# These are the compensation fields — if ANY of these is 0 or missing,
# it is a strong real-world signal that the employee will leave
COMPENSATION_FIELDS = ["MonthlyIncome", "DailyRate", "HourlyRate", "MonthlyRate"]


def common_sense_check(df):
    """
    Apply real-world logic BEFORE the ML model.
    If any compensation is 0 or missing → Will Leave (100% risk).
    This handles cases the model was never trained on.
    Returns a boolean Series: True = override to Will Leave.
    """
    override = pd.Series([False] * len(df), index=df.index)
    for field in COMPENSATION_FIELDS:
        if field in df.columns:
            val = pd.to_numeric(df[field], errors='coerce').fillna(0)
            override = override | (val <= 0)
    return override


def normalize_values(df):
    """
    Convert all numeric columns to proper numbers.
    Cap any unrealistically high values at the maximum.
    Does NOT set a minimum — 0 stays as 0 (handled by common_sense_check).
    """
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
        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    if "OverTime" in df.columns:
        df["OverTime"] = df["OverTime"].map({"Yes": 1, "No": 0})

    cat_cols = ["BusinessTravel", "Department", "EducationField", "JobRole", "MaritalStatus"]
    for col in cat_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=[col], inplace=True)

    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    df[bool_cols] = df[bool_cols].astype(int)
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return df


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict_single", methods=["POST"])
def predict_single():
    data = request.json
    df_input = pd.DataFrame([data])

    # Step 1 — Common sense check first
    override = common_sense_check(df_input)

    if override.iloc[0]:
        # Zero compensation = definite leave, no need for ML
        return jsonify({
            "prediction": "Will Leave",
            "risk": "98.0%",
            "color": "red",
            "reason": "Zero or missing compensation detected"
        })

    # Step 2 — Normal ML prediction
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
    """
    Single fast request — upload CSV + auto-map columns + predict instantly.
    """
    file = request.files["file"]
    content = file.stream.read().decode("utf-8")
    df_raw = pd.read_csv(io.StringIO(content))
    uploaded_columns = df_raw.columns.tolist()

    # Auto-map columns (case-insensitive match)
    mapping = {}
    for model_col in MODEL_REQUIRED_COLUMNS:
        for uploaded_col in uploaded_columns:
            if uploaded_col.strip().lower() == model_col.lower():
                mapping[model_col] = uploaded_col
                break

    # Build mapped dataframe using model column names
    df_mapped = pd.DataFrame()
    for model_col in MODEL_REQUIRED_COLUMNS:
        uploaded_col = mapping.get(model_col)
        if uploaded_col and uploaded_col in df_raw.columns:
            df_mapped[model_col] = df_raw[uploaded_col].values
        else:
            # Missing column → use average default value
            df_mapped[model_col] = COLUMN_DEFAULTS.get(model_col, 0)

    df_mapped = df_mapped.reset_index(drop=True)

    # Step 1 — Common sense check (vectorized, instant even for 10000 rows)
    override_mask = common_sense_check(df_mapped).values

    # Step 2 — ML prediction for all rows
    df_processed = preprocess(df_mapped)
    predictions = model.predict(df_processed).copy()
    probabilities = model.predict_proba(df_processed)[:, 1].copy()

    # Step 3 — Apply overrides for zero-compensation rows
    predictions[override_mask] = 1
    probabilities[override_mask] = 0.98

    # Build result — show first 3 original columns for identification
    preview_cols = uploaded_columns[:3]
    df_result = df_raw[preview_cols].copy().reset_index(drop=True)
    df_result["Prediction"] = ["Will Leave" if p == 1 else "Will Stay" for p in predictions]
    df_result["Attrition Risk %"] = (probabilities * 100).round(1)

    unmatched = [c for c in MODEL_REQUIRED_COLUMNS if c not in mapping]

    return jsonify({
        "results": df_result.to_dict(orient="records"),
        "columns": preview_cols + ["Prediction", "Attrition Risk %"],
        "total": len(predictions),
        "leaving": int(sum(predictions)),
        "staying": int(len(predictions) - sum(predictions)),
        "auto_matched": len(mapping),
        "unmatched": unmatched,
    })


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
