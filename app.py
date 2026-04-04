"""
AttritionIQ — Employee Attrition Predictor
==========================================
HOW IT WORKS:
1. Load trained ML model from disk
2. When employee data comes in, first apply COMMON SENSE RULES
3. If no rules broken, let ML model predict
4. Return result to frontend

FUNCTIONS:
- get_india_min_wage()     → Fetches India minimum wage threshold
- common_sense_check()     → Catches obviously wrong data
- preprocess()             → Converts raw data into ML-ready format
- predict_single route     → Single employee prediction
- upload_and_predict route → Batch CSV prediction
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import io
import os
import urllib.request
import json

app = Flask(__name__)

# ─────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        FEATURE_COLUMNS = pickle.load(f)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Could not load model: {e}")


# ─────────────────────────────────────────────────────────
# MODEL COLUMNS — what the ML model was trained on
# ─────────────────────────────────────────────────────────
MODEL_COLUMNS = [
    "Age", "BusinessTravel", "DailyRate", "Department", "DistanceFromHome",
    "Education", "EducationField", "EnvironmentSatisfaction", "Gender",
    "HourlyRate", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction",
    "MaritalStatus", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
    "OverTime", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"
]

# Default average values used when a CSV column is missing
DEFAULTS = {
    "Age": 37, "DailyRate": 800, "DistanceFromHome": 9, "Education": 3,
    "EnvironmentSatisfaction": 3, "HourlyRate": 65, "JobInvolvement": 3,
    "JobLevel": 2, "JobSatisfaction": 3, "MonthlyIncome": 6500,
    "MonthlyRate": 14000, "NumCompaniesWorked": 2, "PercentSalaryHike": 15,
    "PerformanceRating": 3, "RelationshipSatisfaction": 3, "StockOptionLevel": 1,
    "TotalWorkingYears": 11, "TrainingTimesLastYear": 3, "WorkLifeBalance": 3,
    "YearsAtCompany": 7, "YearsInCurrentRole": 4, "YearsSinceLastPromotion": 2,
    "YearsWithCurrManager": 4,
}


# ─────────────────────────────────────────────────────────
# INDIA MINIMUM WAGE
#
# Source: India Ministry of Labour & Employment
# Central sphere minimum wage for unskilled workers (2024):
# Rs. 176/day × 26 working days = Rs. 4,576/month
#
# Exchange rate fetched live from exchangerate-api.com (free, no key needed)
# Converts INR to USD so it works with our USD-based model
# ─────────────────────────────────────────────────────────
_min_wage_cache = None

def get_india_min_wage():
    """
    Returns India minimum monthly wage in USD.
    India floor wage: Rs.176/day × 26 days = Rs.4,576/month
    Converted to USD using live exchange rate.
    Falls back to Rs.4,576 ÷ 83 = ~$55 if API unavailable.
    """
    global _min_wage_cache
    if _min_wage_cache is not None:
        return _min_wage_cache

    india_min_inr = 176 * 26  # Rs. 4,576/month

    try:
        url = "https://api.exchangerate-api.com/v4/latest/USD"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        response = urllib.request.urlopen(req, timeout=5)
        rates = json.loads(response.read().decode("utf-8"))["rates"]
        usd_to_inr = rates.get("INR", 83)
        min_usd = round(india_min_inr / usd_to_inr, 2)
        _min_wage_cache = {
            "min_inr": india_min_inr,
            "min_usd": min_usd,
            "exchange_rate": round(usd_to_inr, 2),
            "source": "India Ministry of Labour (Rs.176/day × 26 days)",
            "rate_source": "exchangerate-api.com (live)"
        }
    except Exception:
        _min_wage_cache = {
            "min_inr": india_min_inr,
            "min_usd": round(india_min_inr / 83, 2),
            "exchange_rate": 83,
            "source": "India Ministry of Labour (Rs.176/day × 26 days)",
            "rate_source": "Fixed fallback rate"
        }

    return _min_wage_cache


# ─────────────────────────────────────────────────────────
# COMMON SENSE RULES
#
# These run BEFORE the ML model.
# If data is logically impossible or obviously extreme,
# we override with "Will Leave" directly.
#
# RULES:
# 1. Income below India minimum wage → leaving
# 2. Zero daily rate → leaving
# 3. Experience > Age - 14 → impossible data
# 4. Company years > Total experience → impossible
# 5. Role years > Company years → impossible
# 6. Manager years > Company years → impossible
# 7. All satisfaction = 1 + overtime = Yes → extreme burnout → leaving
# ─────────────────────────────────────────────────────────
def common_sense_check(df):
    """
    Returns (override list, reasons list) for each row.
    True in override means force Will Leave.
    """
    n = len(df)
    override = [False] * n
    reasons  = [""] * n

    wage_info    = get_india_min_wage()
    MIN_WAGE_USD = wage_info["min_usd"]

    def get_col(col, default=0):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default).values
        return [default] * n

    income    = get_col("MonthlyIncome")
    daily     = get_col("DailyRate")
    age       = get_col("Age", 25)
    total_exp = get_col("TotalWorkingYears")
    co_yrs    = get_col("YearsAtCompany")
    role_yrs  = get_col("YearsInCurrentRole")
    mgr_yrs   = get_col("YearsWithCurrManager")
    env_sat   = get_col("EnvironmentSatisfaction", 3)
    job_sat   = get_col("JobSatisfaction", 3)
    wlb       = get_col("WorkLifeBalance", 3)

    overtime = ["no"] * n
    if "OverTime" in df.columns:
        overtime = df["OverTime"].astype(str).str.strip().str.lower().values

    for i in range(n):
        # Rule 1: Income below India minimum wage
        if income[i] < MIN_WAGE_USD:
            override[i] = True
            reasons[i]  = f"Income below India minimum wage (₹{wage_info['min_inr']:,}/mo = ~${MIN_WAGE_USD}/mo)"

        # Rule 2: Zero daily rate
        elif daily[i] <= 0:
            override[i] = True
            reasons[i]  = "Zero daily rate — no realistic compensation"

        # Rule 3: Impossible age vs experience
        elif total_exp[i] > 0 and (total_exp[i] + 14) > age[i]:
            override[i] = True
            reasons[i]  = f"Impossible: {int(total_exp[i])} working years but age is only {int(age[i])}"

        # Rule 4: Company years > total career
        elif co_yrs[i] > total_exp[i]:
            override[i] = True
            reasons[i]  = f"Impossible: {int(co_yrs[i])} years at company > {int(total_exp[i])} total experience"

        # Rule 5: Role years > company years
        elif role_yrs[i] > co_yrs[i]:
            override[i] = True
            reasons[i]  = f"Impossible: {int(role_yrs[i])} years in role > {int(co_yrs[i])} years at company"

        # Rule 6: Manager years > company years
        elif mgr_yrs[i] > co_yrs[i]:
            override[i] = True
            reasons[i]  = f"Impossible: {int(mgr_yrs[i])} years with manager > {int(co_yrs[i])} years at company"

        # Rule 7: Extreme burnout
        elif (env_sat[i] <= 1 and job_sat[i] <= 1 and
              wlb[i] <= 1 and overtime[i] == "yes"):
            override[i] = True
            reasons[i]  = "Extreme burnout: all satisfaction at minimum + forced overtime"

    return override, reasons


# ─────────────────────────────────────────────────────────
# PREPROCESS
#
# Converts raw employee data into format the ML model understands.
# ML models only understand numbers, not text like "Male" or "Yes".
# ─────────────────────────────────────────────────────────
def preprocess(df_raw):
    """
    1. Remove irrelevant columns
    2. Convert Gender/OverTime to 0/1
    3. One-hot encode categorical columns
    4. Fill missing columns with 0
    5. Align to exact feature order the model was trained on
    """
    df = df_raw.copy()

    # Remove columns the model doesn't need
    for col in ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours",
                "Attrition", "EmployeeName", "Name"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Convert text to 0/1
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0}).fillna(0)
    if "OverTime" in df.columns:
        df["OverTime"] = df["OverTime"].map({"Yes": 1, "No": 0}).fillna(0)

    # One-hot encode: create separate 0/1 column for each category value
    for col in ["BusinessTravel", "Department", "EducationField", "JobRole", "MaritalStatus"]:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=[col], inplace=True)

    # Convert boolean columns to integers
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    df[bool_cols] = df[bool_cols].astype(int)

    # Align to exact columns model expects (fill any missing with 0)
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return df


# ─────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get_wage_info")
def get_wage_info():
    """Returns India minimum wage info to the frontend."""
    return jsonify(get_india_min_wage())


@app.route("/predict_single", methods=["POST"])
def predict_single():
    """
    Single employee prediction.
    1. Receive data from frontend
    2. Run common sense check
    3. If passes, run ML model
    4. Return prediction
    """
    data = request.json

    # If not overtime, use default hourly rate (irrelevant when no overtime)
    if str(data.get("OverTime", "No")).strip().lower() in ["no", "0"]:
        data["HourlyRate"]  = DEFAULTS["HourlyRate"]
        data["MonthlyRate"] = DEFAULTS["MonthlyRate"]

    # Fill fields not in the form with defaults so model gets all columns
    for col, val in DEFAULTS.items():
        if col not in data:
            data[col] = val

    df = pd.DataFrame([data])

    # Common sense check first
    override, reasons = common_sense_check(df)

    if override[0]:
        return jsonify({
            "prediction": "Will Leave",
            "risk": "98.0%",
            "color": "red",
            "reason": reasons[0],
            "method": "rule"
        })

    # ML model prediction
    df_processed = preprocess(df)
    prediction   = model.predict(df_processed)[0]
    probability  = model.predict_proba(df_processed)[0][1]

    return jsonify({
        "prediction": "Will Leave" if prediction == 1 else "Will Stay",
        "risk": f"{probability * 100:.1f}%",
        "color": "red" if prediction == 1 else "green",
        "reason": "",
        "method": "model"
    })


@app.route("/upload_and_predict", methods=["POST"])
def upload_and_predict():
    """
    Batch prediction for entire CSV file.
    1. Read CSV
    2. Auto-map columns by name matching
    3. Run common sense checks on all rows
    4. Run ML model on all rows
    5. Return results with Employee name, Prediction, Risk, Flag
    """
    file    = request.files["file"]
    content = file.stream.read().decode("utf-8")
    df_raw  = pd.read_csv(io.StringIO(content))
    all_cols = df_raw.columns.tolist()

    # Find employee name column
    name_col = None
    for c in all_cols:
        if any(kw in c.lower() for kw in ["name", "emp_id", "employeeid", "staff_id", "employee"]):
            name_col = c
            break

    # Auto-map uploaded columns to model columns (case-insensitive)
    mapping = {}
    for model_col in MODEL_COLUMNS:
        for up_col in all_cols:
            if up_col.strip().lower() == model_col.lower():
                mapping[model_col] = up_col
                break

    # Build dataframe aligned to model columns
    df_mapped = pd.DataFrame()
    for model_col in MODEL_COLUMNS:
        up_col = mapping.get(model_col)
        if up_col and up_col in df_raw.columns:
            df_mapped[model_col] = df_raw[up_col].values
        else:
            df_mapped[model_col] = DEFAULTS.get(model_col, 0)
    df_mapped = df_mapped.reset_index(drop=True)

    # Run common sense checks
    override_list, reasons_list = common_sense_check(df_mapped)

    # Run ML model on all rows at once
    df_processed  = preprocess(df_mapped)
    predictions   = model.predict(df_processed).astype(int).tolist()
    probabilities = model.predict_proba(df_processed)[:, 1].tolist()

    # Apply overrides
    for i in range(len(predictions)):
        if override_list[i]:
            predictions[i]   = 1
            probabilities[i] = 0.98

    # Build results — Employee name, Prediction, Risk, Flag only
    results = []
    for i in range(len(predictions)):
        row = {}
        if name_col and name_col in df_raw.columns:
            row["Employee"] = str(df_raw[name_col].iloc[i])
        else:
            row["Employee"] = f"Employee {i + 1}"
        row["Prediction"]       = "Will Leave" if predictions[i] == 1 else "Will Stay"
        row["Attrition Risk %"] = round(probabilities[i] * 100, 1)
        row["Flag"]             = reasons_list[i] if override_list[i] else ""
        results.append(row)

    unmatched = [c for c in MODEL_COLUMNS if c not in mapping]

    return jsonify({
        "results":      results,
        "columns":      ["Employee", "Prediction", "Attrition Risk %", "Flag"],
        "total":        len(predictions),
        "leaving":      sum(predictions),
        "staying":      len(predictions) - sum(predictions),
        "auto_matched": len(mapping),
        "unmatched":    unmatched,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
