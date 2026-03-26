# ============================================================
#  STEP 1: Run this ONCE to save your trained model
#  It saves: model.pkl + feature_columns.pkl
# ============================================================

import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# --- Load & clean ---
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
df.drop(columns=["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"], inplace=True)

# --- Binary encoding ---
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
df["Gender"]    = df["Gender"].map({"Male": 1, "Female": 0})
df["OverTime"]  = df["OverTime"].map({"Yes": 1, "No": 0})

# --- One-hot encoding ---
cat_cols = ["BusinessTravel", "Department", "EducationField", "JobRole", "MaritalStatus"]
for col in cat_cols:
    dummies = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummies], axis=1)
    df.drop(columns=[col], inplace=True)

bool_cols = df.select_dtypes(include="bool").columns.tolist()
df[bool_cols] = df[bool_cols].astype(int)

# --- Train ---
X = df.drop(columns=["Attrition"])
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_jobs=-1, random_state=42)
clf.fit(X_train, y_train)

acc = accuracy_score(y_test, clf.predict(X_test))
print(f"Model Accuracy: {acc * 100:.2f}%")

# --- Save everything in one dictionary for app.py ---
model_data = {
    'model': clf,
    'columns': X.columns.tolist()
}

with open("attrition_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("✅ SUCCESS: attrition_model.pkl created!")
print("Now your app.py will work without any KeyErrors.")