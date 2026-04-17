"""
============================================================
Food Demand Forecasting as a Proxy for Food Waste Prediction
Machine Learning Pipeline — COM 572 Coursework Task 1
============================================================
"""

# ─────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────
# 2. LOAD AND MERGE DATASETS
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading and Merging Datasets")
print("=" * 60)

train     = pd.read_csv("train.csv")
meal_info = pd.read_csv("meal_info.csv")
center    = pd.read_csv("fulfilment_center_info.csv")

# Merge on shared keys
df = (
    train
    .merge(meal_info, on="meal_id", how="left")
    .merge(center,    on="center_id", how="left")
)

print(f"Merged dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# ─────────────────────────────────────────────
# 3. CLEAN COLUMN NAMES
# ─────────────────────────────────────────────
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# ─────────────────────────────────────────────
# 4. MISSING VALUES AND DUPLICATES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Data Quality Checks")
print("=" * 60)

print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nDuplicate rows: {df.duplicated().sum()}")
print(f"\nDataset summary:\n{df.describe().round(2)}")

# ─────────────────────────────────────────────
# 5. CREATE DERIVED WASTE TARGET
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Creating Derived Waste Target")
print("=" * 60)

# RATIONALE: The dataset contains no direct food waste labels.
# We use num_orders as a demand proxy: weeks with demand below
# the median are more likely to produce unsold surplus (High Waste),
# while high-demand weeks are associated with Low Waste.
# This is a well-documented proxy strategy in supply chain
# waste literature [1][2].

median_orders = df["num_orders"].median()
print(f"Median orders (threshold): {median_orders}")

# Binary target: 1 = High Waste (low demand), 0 = Low Waste (high demand)
df["waste_label"] = (df["num_orders"] <= median_orders).astype(int)

print(f"\nClass distribution:")
vc = df["waste_label"].value_counts()
print(f"  Low Waste  (0): {vc[0]:,}  ({vc[0]/len(df)*100:.1f}%)")
print(f"  High Waste (1): {vc[1]:,}  ({vc[1]/len(df)*100:.1f}%)")

# ─────────────────────────────────────────────
# 6. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Exploratory Data Analysis")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Exploratory Data Analysis — Food Demand & Waste Proxy", fontsize=15, y=1.01)

# 6a. Distribution of num_orders
axes[0, 0].hist(df["num_orders"], bins=60, color="#2E86AB", edgecolor="white", alpha=0.85)
axes[0, 0].axvline(median_orders, color="#E84855", linestyle="--", linewidth=2,
                   label=f"Median = {int(median_orders)}")
axes[0, 0].set_title("Distribution of Orders (num_orders)")
axes[0, 0].set_xlabel("Number of Orders")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].legend()

# 6b. Waste label class balance
labels = ["Low Waste (0)", "High Waste (1)"]
counts = [vc[0], vc[1]]
axes[0, 1].bar(labels, counts, color=["#44BBA4", "#E84855"], edgecolor="white", width=0.5)
axes[0, 1].set_title("Waste Label Class Distribution")
axes[0, 1].set_ylabel("Count")
for i, v in enumerate(counts):
    axes[0, 1].text(i, v + 2000, f"{v:,}", ha="center", fontsize=10)

# 6c. Orders by cuisine
cuisine_order = df.groupby("cuisine")["num_orders"].median().sort_values()
axes[0, 2].barh(cuisine_order.index, cuisine_order.values, color="#F18F01", edgecolor="white")
axes[0, 2].set_title("Median Orders by Cuisine")
axes[0, 2].set_xlabel("Median Orders")

# 6d. Orders by category (top 10)
cat_order = df.groupby("category")["num_orders"].median().sort_values(ascending=False).head(10)
axes[1, 0].barh(cat_order.index[::-1], cat_order.values[::-1], color="#A23B72", edgecolor="white")
axes[1, 0].set_title("Median Orders by Category (Top 10)")
axes[1, 0].set_xlabel("Median Orders")

# 6e. Orders by center type
ct = df.groupby("center_type")["num_orders"].median()
axes[1, 1].bar(ct.index, ct.values, color=["#2E86AB", "#44BBA4", "#F18F01"], edgecolor="white", width=0.5)
axes[1, 1].set_title("Median Orders by Centre Type")
axes[1, 1].set_ylabel("Median Orders")

# 6f. Checkout vs base price scatter (sampled)
sample = df.sample(5000, random_state=42)
sc = axes[1, 2].scatter(sample["base_price"], sample["checkout_price"],
                         c=sample["waste_label"], cmap="RdYlGn_r", alpha=0.4, s=8)
axes[1, 2].set_title("Checkout vs Base Price by Waste Label")
axes[1, 2].set_xlabel("Base Price (£)")
axes[1, 2].set_ylabel("Checkout Price (£)")
plt.colorbar(sc, ax=axes[1, 2], label="1=High Waste")

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("EDA plots saved to: eda_plots.png")

# ─────────────────────────────────────────────
# 7. FEATURE SELECTION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Feature Selection")
print("=" * 60)

# Per brief: exclude 'id' and 'num_orders'; use only specified features
FEATURES = [
    "week", "center_id", "meal_id",
    "checkout_price", "base_price",
    "emailer_for_promotion", "homepage_featured",
    "category", "cuisine",
    "city_code", "region_code",
    "center_type", "op_area"
]

TARGET = "waste_label"

X = df[FEATURES].copy()
y = df[TARGET].copy()

# Add engineered feature: price discount ratio
X["discount_ratio"] = ((X["base_price"] - X["checkout_price"]) / X["base_price"]).clip(lower=0)

# Update feature lists after engineering
numerical_features = [
    "week", "checkout_price", "base_price", "op_area",
    "discount_ratio"
]
categorical_features = [
    "center_id", "meal_id", "city_code", "region_code",
    "emailer_for_promotion", "homepage_featured",
    "category", "cuisine", "center_type"
]

print(f"Numerical features : {numerical_features}")
print(f"Categorical features: {categorical_features}")
print(f"Total features used : {len(numerical_features) + len(categorical_features)}")

# ─────────────────────────────────────────────
# 8. TRAIN-TEST SPLIT (STRATIFIED)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size : {X_train.shape[0]:,}")
print(f"Test  size : {X_test.shape[0]:,}")

# ─────────────────────────────────────────────
# 9. PREPROCESSING PIPELINE
# ─────────────────────────────────────────────
# Numerical: StandardScaler
# Categorical: OneHotEncoder (handle_unknown='ignore' for robustness)
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
])

# ─────────────────────────────────────────────
# 10. MODEL DEFINITIONS
# ─────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, C=1.0, solver="lbfgs"
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=10, min_samples_split=20, min_samples_leaf=10, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    )
}

pipelines = {
    name: Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   clf)
    ])
    for name, clf in models.items()
}

# ─────────────────────────────────────────────
# 11. TRAINING AND EVALUATION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Model Training and Evaluation")
print("=" * 60)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
results = {}

for name, pipe in pipelines.items():
    print(f"\n  Training: {name}")
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    # 3-fold cross-validation on F1
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)

    results[name] = {
        "Accuracy":  round(acc,  4),
        "Precision": round(prec, 4),
        "Recall":    round(rec,  4),
        "F1-Score":  round(f1,   4),
        "CV F1 Mean": round(cv_scores.mean(), 4),
        "CV F1 Std":  round(cv_scores.std(),  4),
        "y_pred":    y_pred,
        "pipe":      pipe
    }

    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1-Score  : {f1:.4f}")
    print(f"    CV F1     : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────────
# 12. CONFUSION MATRICES
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Confusion Matrices — All Models", fontsize=14)

for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Low Waste", "High Waste"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=12)

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nConfusion matrices saved to: confusion_matrices.png")

# ─────────────────────────────────────────────
# 13. MODEL COMPARISON CHART
# ─────────────────────────────────────────────
metrics_df = pd.DataFrame({
    name: {k: v for k, v in res.items()
           if k in ["Accuracy", "Precision", "Recall", "F1-Score"]}
    for name, res in results.items()
}).T

print("\n" + "=" * 60)
print("STEP 7: Model Comparison")
print("=" * 60)
print(metrics_df.to_string())

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(metrics_df))
width = 0.2
colours = ["#2E86AB", "#44BBA4", "#E84855", "#F18F01"]

for i, col in enumerate(["Accuracy", "Precision", "Recall", "F1-Score"]):
    ax.bar(x + i * width, metrics_df[col], width, label=col, color=colours[i], edgecolor="white")

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metrics_df.index, fontsize=11)
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Model comparison chart saved to: model_comparison.png")

# ─────────────────────────────────────────────
# 14. HYPERPARAMETER TUNING — RANDOM FOREST
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: GridSearchCV — Random Forest Optimisation")
print("=" * 60)

param_grid = {
    "classifier__n_estimators":    [100, 150],
    "classifier__max_depth":       [10, 12],
    "classifier__min_samples_split": [10],
    "classifier__min_samples_leaf":  [5],
}

rf_base_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",   RandomForestClassifier(random_state=42, n_jobs=-1))
])

grid_search = GridSearchCV(
    estimator=rf_base_pipe,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\nBest parameters : {grid_search.best_params_}")
print(f"Best CV F1 score: {grid_search.best_score_:.4f}")

# Evaluate tuned model
best_rf = grid_search.best_estimator_
y_pred_rf_tuned = best_rf.predict(X_test)

tuned_f1   = f1_score(y_test, y_pred_rf_tuned)
tuned_acc  = accuracy_score(y_test, y_pred_rf_tuned)
tuned_prec = precision_score(y_test, y_pred_rf_tuned)
tuned_rec  = recall_score(y_test, y_pred_rf_tuned)

print(f"\nTuned Random Forest on Test Set:")
print(f"  Accuracy  : {tuned_acc:.4f}")
print(f"  Precision : {tuned_prec:.4f}")
print(f"  Recall    : {tuned_rec:.4f}")
print(f"  F1-Score  : {tuned_f1:.4f}")
print(f"\nClassification Report:\n")
print(classification_report(y_test, y_pred_rf_tuned,
                             target_names=["Low Waste", "High Waste"]))

# ─────────────────────────────────────────────
# 15. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: Feature Importance (Tuned Random Forest)")
print("=" * 60)

# Extract feature names from the fitted pipeline
ohe = best_rf.named_steps["preprocessor"].named_transformers_["cat"]
cat_feature_names = ohe.get_feature_names_out(categorical_features).tolist()
all_feature_names = numerical_features + cat_feature_names

importances = best_rf.named_steps["classifier"].feature_importances_
feat_imp_df = pd.DataFrame({
    "Feature": all_feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False).head(20)

print(feat_imp_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(feat_imp_df["Feature"][::-1], feat_imp_df["Importance"][::-1],
        color="#2E86AB", edgecolor="white")
ax.set_title("Top 20 Feature Importances — Tuned Random Forest", fontsize=13)
ax.set_xlabel("Gini Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Feature importance chart saved to: feature_importance.png")

# ─────────────────────────────────────────────
# 16. SAVE FINAL MODEL
# ─────────────────────────────────────────────
joblib.dump(best_rf, "best_rf_model.pkl")
print("\nFinal model saved to: best_rf_model.pkl")

# ─────────────────────────────────────────────
# 17. FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"""
┌────────────────────────────────────────────────────────────────────────┐
│  Model                 │ Accuracy │ Precision │  Recall  │  F1-Score  │
├────────────────────────────────────────────────────────────────────────┤
│  Logistic Regression   │ {results['Logistic Regression']['Accuracy']:.4f}   │  {results['Logistic Regression']['Precision']:.4f}   │  {results['Logistic Regression']['Recall']:.4f}  │  {results['Logistic Regression']['F1-Score']:.4f}    │
│  Decision Tree         │ {results['Decision Tree']['Accuracy']:.4f}   │  {results['Decision Tree']['Precision']:.4f}   │  {results['Decision Tree']['Recall']:.4f}  │  {results['Decision Tree']['F1-Score']:.4f}    │
│  Random Forest (base)  │ {results['Random Forest']['Accuracy']:.4f}   │  {results['Random Forest']['Precision']:.4f}   │  {results['Random Forest']['Recall']:.4f}  │  {results['Random Forest']['F1-Score']:.4f}    │
│  Random Forest (tuned) │ {tuned_acc:.4f}   │  {tuned_prec:.4f}   │  {tuned_rec:.4f}  │  {tuned_f1:.4f}    │
└────────────────────────────────────────────────────────────────────────┘

  ✔ Final model: Tuned Random Forest
  ✔ Target was constructed as a proxy (NOT real waste labels)
  ✔ All plots saved to current directory
  ✔ Model saved to: best_rf_model.pkl
""")

# ─────────────────────────────────────────────
# WHY NOT OTHER MODELS?
# ─────────────────────────────────────────────
"""
Model Justification Summary
─────────────────────────────
USED:
  Logistic Regression  — Interpretable linear baseline; fast to train; meaningful
                         coefficient weights for stakeholder explainability [3].
  Decision Tree        — Transparent rule-based model; risk of overfitting without
                         depth constraints; valuable for visualising decision logic [4].
  Random Forest        — Ensemble of trees; reduces overfitting via bagging;
                         strong generalisation on tabular data [5].

NOT USED:
  SVM                  — Computationally prohibitive on 456 K+ samples; does not
                         scale well with RBF kernel without sampling strategies [6].
  KNN                  — O(n) prediction time; memory-intensive; poor scaling to
                         large, high-dimensional datasets post-OHE [6].
  Naïve Bayes          — Assumes conditional feature independence; violated by
                         correlated pricing and promotional variables [7].
  Neural Networks       — Require substantial tuning, data, and compute; limited
                         interpretability for stakeholder reporting [8].
  Gradient Boosting /  — Strong performance but computationally expensive for
  XGBoost              — this scope; parameter sensitivity demands large search
                         grids incompatible with the brief constraints [9].
"""
