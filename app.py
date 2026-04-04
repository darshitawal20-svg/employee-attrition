from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import io
import os
import urllib.request
import json

app = Flask(__name__)

# ─────────────────────────────────────────────────────────
# STEP 1: LOAD MODEL & FEATURES
# ─────────────────────────────────────────────────────────
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        FEATURE_COLUMNS = pickle.load(f)
    print("✅ Model & Logic Synchronized.")
except Exception as e:
    print(f"❌ Load Error: {e}")

# Default values for fields not in your "Single" form cards
DEFAULTS = {
    "DistanceFromHome": 9, "BusinessTravel": "Travel_Rarely",
    "PercentSalaryHike": 15, "TrainingTimesLastYear": 3,
    "YearsWithCurrManager": 4, "JobInvolvement": 3,
    "RelationshipSatisfaction": 3, "EducationField": "Life Sciences"
}

# ─────────────────────────────────────────────────────────
# STEP 2: COMMON SENSE RULES (The Intelligence Layer)
# ─────────────────────────────────────────────────────────
def common_sense_check(df):
    n = len(df)
    override = [False] * n
    reasons = [""] * n
    
    # We use $55 as the India Minimum Wage threshold (~Rs. 4,576)
    MIN_MONTHLY_INCOME = 55.0 

    for i in range(n):
        row = df.iloc[i]
        income = pd.to_numeric(row.get("MonthlyIncome", 0))
        daily  = pd.to_numeric(row.get("DailyRate", 0))
        age    = pd.to_numeric(row.get("Age", 30))
        exp    = pd.to_numeric(row.get("TotalWorkingYears", 0))
        co     = pd.to_numeric(row.get("YearsAtCompany", 0))
        role   = pd.to_numeric(row.get("YearsInCurrentRole", 0))

        # RULE 1: THE POVERTY CHECK (Solves your $1 Daily Rate issue)
        # We flag anything under $100 daily as "Unrealistic"
        if daily < 100 or income < MIN_MONTHLY_INCOME:
            override[i] = True
            reasons[i] = "Unrealistic Pay: Below survival threshold for this role."
        
        # RULE 2: THE BIOLOGICAL PARADOX
        elif exp + 14 > age:
            override[i] = True
            reasons[i] = "Logic Error: Total Experience is impossible for this age."

        # RULE 3: THE TENURE PARADOX
        elif co > exp:
            override[i] = True
            reasons[i] = "Logic Error: Years at Company cannot exceed Total Career."
            
        # RULE 4: THE ROLE PARADOX
        elif role > co:
            override[i] = True
            reasons[i] = "Logic Error: Years in Role cannot exceed time at Company."

    return override, reasons

# ─────────────────────────────────────────────────────────
# STEP 3: DATA PREPARATION
# ─────────────────────────────────────────────────────────
def preprocess(df_raw):
    df = df_raw.copy()
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0}).fillna(0)
    if "OverTime" in df.columns:
        df["OverTime"] = df["OverTime"].map({"Yes": 1, "No": 0}).fillna(0)

    for col in ["BusinessTravel", "Department", "EducationField", "JobRole", "MaritalStatus"]:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1).drop(columns=[col])

    return df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

# ─────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict_single", methods=["POST"])
def predict_single():
    data = request.json
    # Fill hidden fields so model doesn't crash
    for col, val in DEFAULTS.items():
        if col not in data: data[col] = val
        
    df = pd.DataFrame([data])
    
    # 1. RUN INTELLIGENCE CHECK FIRST
    override, reasons = common_sense_check(df)
    if override[0]:
        return jsonify({
            "prediction": "Will Leave", 
            "risk": "99.0%", 
            "color": "red",
            "reason": reasons[0], 
            "method": "rule" # Triggers the Yellow Tag
        })

    # 2. RUN AI MODEL SECOND
    df_processed = preprocess(df)
    prob = model.predict_proba(df_processed)[0][1]

    return jsonify({
        "prediction": "Will Leave" if prob > 0.5 else "Will Stay",
        "risk": f"{prob * 100:.1f}%",
        "color": "red" if prob > 0.5 else "green",
        "method": "model" # Triggers the Blue Tag
    })

@app.route("/upload_and_predict", methods=["POST"])
def upload_and_predict():
    file = request.files["file"]
    df_raw = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
    ov, re = common_sense_check(df_raw)
    probs = model.predict_proba(preprocess(df_raw))[:, 1]

    results = []
    for i in range(len(probs)):
        results.append({
            "Employee": df_raw.iloc[i].get("EmpName", f"EMP_{i+1}"),
            "Prediction": "Will Leave" if (probs[i] > 0.5 or ov[i]) else "Will Stay",
            "Attrition Risk %": 99.0 if ov[i] else round(probs[i]*100, 1),
            "Flag": re[i] if ov[i] else ""
        })

    return jsonify({"results": results, "columns": ["Employee", "Prediction", "Attrition Risk %", "Flag"], "total": len(results), "leaving": sum(1 for r in results if r["Prediction"]=="Will Leave"), "staying": sum(1 for r in results if r["Prediction"]=="Will Stay")})

if __name__ == "__main__":
    app.run(debug=True)
