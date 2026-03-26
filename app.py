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


def preprocess(df_raw):
    df = df_raw.copy()
    for col in ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    if "Attrition" in df.columns:
        df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
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
    if "Attrition" in df.columns:
        df.drop(columns=["Attrition"], inplace=True)
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
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
