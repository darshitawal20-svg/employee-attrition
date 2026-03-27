from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load Model
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        FEATURE_COLUMNS = pickle.load(f)
except Exception as e:
    print(f"CRITICAL ERROR: {e}")

# --- THE MASTER INTELLIGENCE LAYER ---
def get_logic_failure_mask(df):
    """
    Validates every relationship in the data.
    If it's physically impossible or unrealistic, it triggers a 'Will Leave' override.
    """
    # Convert to numeric for math
    age = pd.to_numeric(df.get("Age", 0), errors='coerce').fillna(0)
    total_exp = pd.to_numeric(df.get("TotalWorkingYears", 0), errors='coerce').fillna(0)
    company_yrs = pd.to_numeric(df.get("YearsAtCompany", 0), errors='coerce').fillna(0)
    role_yrs = pd.to_numeric(df.get("YearsInCurrentRole", 0), errors='coerce').fillna(0)
    manager_yrs = pd.to_numeric(df.get("YearsWithCurrManager", 0), errors='coerce').fillna(0)
    promo_yrs = pd.to_numeric(df.get("YearsSinceLastPromotion", 0), errors='coerce').fillna(0)
    income = pd.to_numeric(df.get("MonthlyIncome", 0), errors='coerce').fillna(0)
    daily = pd.to_numeric(df.get("DailyRate", 0), errors='coerce').fillna(0)
    
    # CRITICAL RULES:
    rule_bio = (total_exp + 14 > age) # Experience cannot start before age 14
    rule_tenure = (company_yrs > total_exp) # Cannot work at company longer than total career
    rule_internal = (role_yrs > company_yrs) | (manager_yrs > company_yrs) | (promo_yrs > company_yrs)
    rule_poverty = (income < 500) | (daily < 100) # Unrealistic pay floor

    return rule_bio | rule_tenure | rule_internal | rule_poverty

def preprocess(df_raw):
    df = df_raw.copy()
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

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict_single", methods=["POST"])
def predict_single():
    data = request.json
    df_input = pd.DataFrame([data])
    
    # 1. Intelligence Layer Check
    if get_logic_failure_mask(df_input).iloc[0]:
        return jsonify({
            "prediction": "Will Leave", 
            "risk": "100.0%", 
            "color": "red", 
            "reason": "CRITICAL LOGIC ERROR: Impossible data relationships detected."
        })
    
    # 2. ML Prediction
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
    file = request.files["file"]
    df_raw = pd.read_csv(file)
    df_processed = preprocess(df_raw)
    predictions = model.predict(df_processed)
    probs = model.predict_proba(df_processed)[:, 1]
    
    mask = get_logic_failure_mask(df_raw).values
    predictions[mask] = 1
    probs[mask] = 1.0
    
    res = df_raw.iloc[:, :3].copy()
    res["Prediction"] = ["Will Leave" if p == 1 else "Will Stay" for p in predictions]
    res["Risk %"] = (probs * 100).round(1)
    return jsonify({"results": res.to_dict(orient="records"), "columns": res.columns.tolist(), "total": len(predictions), "leaving": int(sum(predictions)), "staying": int(len(predictions) - sum(predictions))})

if __name__ == "__main__":
    app.run(debug=True)
