# ============================================================
#  Employee Attrition Prediction  |  NeuralNine Methodology
#  Uses: pandas, matplotlib, scikit-learn
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────────────────
print("=" * 60)
print("  EMPLOYEE ATTRITION PREDICTION")
print("=" * 60)

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

print(f"\n[1] Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print("\nFirst 3 rows:")
print(df.head(3).to_string())

print("\nColumn dtypes:")
print(df.dtypes.to_string())

print(f"\nMissing values: {df.isnull().sum().sum()}")


# ──────────────────────────────────────────────────────────
# 2. DATA EXPLORATION & CLEANING
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  [2] DATA CLEANING")
print("=" * 60)

# Drop irrelevant columns
irrelevant_cols = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]
df.drop(columns=irrelevant_cols, inplace=True)
print(f"\nDropped irrelevant columns: {irrelevant_cols}")
print(f"Remaining columns: {df.shape[1]}")


# ──────────────────────────────────────────────────────────
# 3. PREPROCESSING & ENCODING
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  [3] PREPROCESSING & ENCODING")
print("=" * 60)

# --- Binary Encoding ---
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
df["Gender"]    = df["Gender"].map({"Male": 1, "Female": 0})
df["OverTime"]  = df["OverTime"].map({"Yes": 1, "No": 0})
print("\nBinary encoded: Attrition, Gender, OverTime")

# --- One-Hot Encoding ---
cat_cols = ["BusinessTravel", "Department", "EducationField", "JobRole", "MaritalStatus"]

for col in cat_cols:
    dummies = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummies], axis=1)
    df.drop(columns=[col], inplace=True)

print(f"One-hot encoded: {cat_cols}")

# --- Convert booleans to int ---
bool_cols = df.select_dtypes(include="bool").columns.tolist()
df[bool_cols] = df[bool_cols].astype(int)
print(f"Converted {len(bool_cols)} boolean columns to int")

print(f"\nFinal dataset shape after encoding: {df.shape}")
print("\nAll feature names:")
for i, c in enumerate(df.columns, 1):
    print(f"  {i:>2}. {c}")


# ──────────────────────────────────────────────────────────
# 4. DATA VISUALIZATION  —  Histogram Grid
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  [4] DATA VISUALIZATION")
print("=" * 60)

cols = df.columns.tolist()
n_cols = 6
n_rows = -(-len(cols) // n_cols)  # ceiling division

fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, n_rows * 2.8))
fig.suptitle("Employee Attrition – Feature Distributions",
             fontsize=16, fontweight="bold", y=1.01)
fig.patch.set_facecolor("#0f1117")

for ax in axes.flat:
    ax.set_facecolor("#1a1d27")
    ax.tick_params(colors="#aaaaaa", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")

for i, col in enumerate(cols):
    ax = axes.flat[i]
    ax.hist(df[col].dropna(), bins=20, color="#5c6bc0", edgecolor="#3a3d5c", linewidth=0.4)
    ax.set_title(col, fontsize=7.5, color="#ccccdd", pad=3, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Count", fontsize=6, color="#888899")

# Hide unused subplots
for j in range(len(cols), len(axes.flat)):
    axes.flat[j].set_visible(False)

plt.tight_layout()
plt.savefig("feature_distributions.png", dpi=150, bbox_inches="tight",
            facecolor="#0f1117")
plt.show()
print("Histogram grid saved → feature_distributions.png")


# ──────────────────────────────────────────────────────────
# 5. MODEL TRAINING
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  [5] MODEL TRAINING")
print("=" * 60)

X = df.drop(columns=["Attrition"])
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"\nTrain size : {len(X_train):,}")
print(f"Test  size : {len(X_test):,}")

clf = RandomForestClassifier(n_jobs=-1, random_state=42)
clf.fit(X_train, y_train)
print("\nRandomForestClassifier trained ✓")


# ──────────────────────────────────────────────────────────
# 6. EVALUATION & FEATURE IMPORTANCE
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  [6] EVALUATION & FEATURE IMPORTANCE")
print("=" * 60)

y_pred   = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n  ★  Model Accuracy on Test Set: {accuracy * 100:.2f}%")

# Feature importances
importances = pd.Series(clf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("\nTop 15 Feature Importances:")
print(importances.head(15).to_string())

# --- Feature Importance Bar Chart ---
top_n     = 20
top_feats = importances.head(top_n).sort_values(ascending=True)

# colour gradient: low → high importance
norm    = plt.Normalize(top_feats.min(), top_feats.max())
cmap    = plt.cm.get_cmap("plasma")
colours = [cmap(norm(v)) for v in top_feats.values]

fig2, ax2 = plt.subplots(figsize=(12, 8))
fig2.patch.set_facecolor("#0f1117")
ax2.set_facecolor("#1a1d27")

bars = ax2.barh(top_feats.index, top_feats.values, color=colours,
                edgecolor="#0f1117", linewidth=0.6, height=0.72)

# Value labels on bars
for bar, val in zip(bars, top_feats.values):
    ax2.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
             f"{val:.4f}", va="center", ha="left",
             fontsize=8, color="#ccccdd")

ax2.set_title(f"Top {top_n} Features Influencing Employee Attrition",
              fontsize=14, fontweight="bold", color="#e8e8ff", pad=14)
ax2.set_xlabel("Feature Importance (Gini)", fontsize=10, color="#aaaacc")
ax2.tick_params(axis="y", labelsize=9, colors="#ccccdd")
ax2.tick_params(axis="x", labelsize=8, colors="#888899")
ax2.set_xlim(0, top_feats.max() * 1.18)

for spine in ax2.spines.values():
    spine.set_edgecolor("#333344")

# Accuracy annotation box
ax2.text(0.98, 0.02, f"Accuracy: {accuracy * 100:.2f}%",
         transform=ax2.transAxes, ha="right", va="bottom",
         fontsize=11, color="#76ff7a", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e3b1e",
                   edgecolor="#76ff7a", linewidth=1.2))

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight",
            facecolor="#0f1117")
plt.show()
print(f"\nFeature importance chart saved → feature_importance.png")

print("\n" + "=" * 60)
print("  PIPELINE COMPLETE")
print("=" * 60)
