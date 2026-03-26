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

# Min values from IBM HR dataset — anything below gets clamped to minimum
# This means 0 income → treated as 1009 (lowest possible) → model correctly predicts high risk
COLUMN_MINS = {
    "Age": 18,
    "DailyRate": 102,
    "DistanceFromHome": 1,
    "Education": 1,
    "EnvironmentSatisfaction": 1,
    "HourlyRate": 30,
    "JobInvolvement": 1,
    "JobLevel": 1,
    "JobSatisfaction": 1,
    "MonthlyIncome": 0,  # CRITICAL: Changed from 1009 to 0
    "MonthlyRate": 2094,
    "NumCompaniesWorked": 0,
    "OverTime": 0,
    "PercentSalaryHike": 11,
    "PerformanceRating": 1,
    "RelationshipSatisfaction": 1,
    "StockOptionLevel": 0,
    "TotalWorkingYears": 0,
    "TrainingTimesLastYear": 0,
    "WorkLifeBalance": 1,
    "YearsAtCompany": 0,
    "YearsInCurrentRole": 0,
    "YearsSinceLastPromotion": 0,
    "YearsWithCurrManager": 0,
}

COLUMN_MAXS = {
    "Age": 60,
    "DailyRate": 1499,
    "DistanceFromHome": 29,
    "Education": 5,
    "EnvironmentSatisfaction": 4,
    "HourlyRate": 100,
    "JobInvolvement": 4,
    "JobLevel": 5,
    "JobSatisfaction": 4,
    "MonthlyIncome": 19999,
    "MonthlyRate": 26999,
    "NumCompaniesWorked": 9,
    "PercentSalaryHike": 25,
    "PerformanceRating": 4,
    "RelationshipSatisfaction": 4,
    "StockOptionLevel": 3,
    "TotalWorkingYears": 40,
    "TrainingTimesLastYear": 6,
    "WorkLifeBalance": 4,
    "YearsAtCompany": 40,
    "YearsInCurrentRole": 18,
    "YearsSinceLastPromotion": 15,
    "YearsWithCurrManager": 17,
}


def clamp_values(df):
    df = df.copy()
    for col, min_val in COLUMN_MINS.items():
        if col in df.columns:
            # Convert to numeric first
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Fill ONLY empty cells with the minimum. 
            # If the user typed '0', it stays '0'.
            df[col] = df[col].fillna(min_val)
            
            # Apply the floor
            df[col] = df[col].clip(lower=min_val)
            
    for col, max_val in COLUMN_MAXS.items():
        if col in df.columns:
            df[col] = df[col].clip(upper=max_val)
    return df


def preprocess(df_raw):
    df = df_raw.copy()

    # Drop irrelevant columns if present
    for col in ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours", "Attrition"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Clamp all values to realistic ranges BEFORE encoding
    df = clamp_values(df)

    # Binary encoding
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

    # Align to training features
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


@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    file = request.files["file"]
    content = file.stream.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(content))
    uploaded_columns = df.columns.tolist()
    auto_mapping = {}
    for model_col in MODEL_REQUIRED_COLUMNS:
        for uploaded_col in uploaded_columns:
            if uploaded_col.strip().lower() == model_col.lower():
                auto_mapping[model_col] = uploaded_col
                break
    return jsonify({
        "uploaded_columns": uploaded_columns,
        "model_columns": MODEL_REQUIRED_COLUMNS,
        "model_descriptions": MODEL_COLUMN_DESCRIPTIONS,
        "auto_mapping": auto_mapping,
        "csv_content": content,
        "row_count": len(df)
    })


@app.route("/predict_batch_mapped", methods=["POST"])
def predict_batch_mapped():
    data = request.json
    csv_content = data["csv_content"]
    mapping = data["mapping"]
    df_raw = pd.read_csv(io.StringIO(csv_content))
    df_mapped = pd.DataFrame()
    for model_col, uploaded_col in mapping.items():
        if uploaded_col and uploaded_col in df_raw.columns:
            df_mapped[model_col] = df_raw[uploaded_col]
        else:
            df_mapped[model_col] = COLUMN_MINS.get(model_col, 0)
    df_processed = preprocess(df_mapped)
    predictions = model.predict(df_processed)
    probabilities = model.predict_proba(df_processed)[:, 1]
    preview_cols = list(df_raw.columns[:3])
    df_result = df_raw[preview_cols].copy()
    df_result["Prediction"] = ["Will Leave" if p == 1 else "Will Stay" for p in predictions]
    df_result["Attrition Risk %"] = (probabilities * 100).round(1)
    results = df_result.to_dict(orient="records")
    return jsonify({
        "results": results,
        "columns": preview_cols + ["Prediction", "Attrition Risk %"],
        "total": len(predictions),
        "leaving": int(sum(predictions)),
        "staying": int(len(predictions) - sum(predictions))
    })


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
