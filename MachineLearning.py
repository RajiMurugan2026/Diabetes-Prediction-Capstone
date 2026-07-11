import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

# Load Cleaned Dataset
df = pd.read_csv("cleaned_data.csv")

# =====================================================
# Handle Remaining Missing Values for Machine Learning
# =====================================================

df["SkinThickness"] = df["SkinThickness"].fillna(
    df["SkinThickness"].median()
)

print("Remaining Missing Values")
print(df.isnull().sum())

print("=" * 60)
print("Missing values in each column")
print("=" * 60)
print(df.isnull().sum())

print("\nTotal missing values:", df.isnull().sum().sum())

print("="*60)
print("First Five Rows")
print("="*60)

print(df.head())

print("\n")

print("="*60)
print("Dataset Shape")
print("="*60)

print(df.shape)

print("\n")

print("="*60)
print("Data Types")
print("="*60)

print(df.dtypes)
print("=" * 60)
print("Missing Values")
print("=" * 60)

print(df.isnull().sum())

print("\nTotal Missing Values:", df.isnull().sum().sum())
# Features and Labels
# Regression Target
y_reg = df["Glucose"]

# Classification Target
y_clf = df["Outcome"]

# Feature Matrix
X = df.drop(columns=["Glucose", "Outcome"])

print("\nMissing values in X:")
print(X.isnull().sum())

print("\nTotal missing values in X:", X.isnull().sum().sum())

print("="*60)
print("Feature Matrix Shape")
print("="*60)

print(X.shape)

print("\nRegression Label Shape:", y_reg.shape)

print("Classification Label Shape:", y_clf.shape)

# Check Categorical Columns
print("="*60)
print("Categorical Columns")
print("="*60)

categorical_cols = X.select_dtypes(include=["object", "category"]).columns

print(categorical_cols)

# Task 2: Train-Test Split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X,
    y_reg,
    test_size=0.20,
    random_state=42
)

# Classification Split
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X,
    y_clf,
    test_size=0.20,
    random_state=42,
    stratify=y_clf
)

print("=" * 60)
print("Regression Training Shape :", X_train_reg.shape)
print("Regression Testing Shape  :", X_test_reg.shape)

print("\nClassification Training Shape :", X_train_clf.shape)
print("Classification Testing Shape  :", X_test_clf.shape)

# Scale Features for Regression
scaler_reg = StandardScaler()

X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)

X_test_reg_scaled = scaler_reg.transform(X_test_reg)

print("Regression scaling completed.")
# Scale Features for Classification
scaler_clf = StandardScaler()

X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)

X_test_clf_scaled = scaler_clf.transform(X_test_clf)

print("Classification scaling completed.")

# Task 4: Linear Regression
linear_model = LinearRegression()

print("=" * 60)
print("Missing values in DataFrame")
print(df.isnull().sum())

print("\nMissing values in X")
print(X.isnull().sum())

print("\nTotal missing values in X:", X.isnull().sum().sum())

#raise SystemExit("Debug complete")

linear_model.fit(X_train_reg_scaled, y_train_reg)

print("Linear Regression model trained successfully.")
# Predict on Test Data
y_pred_reg = linear_model.predict(X_test_reg_scaled)

print("Prediction completed.")

# Model Evaluation
mse = mean_squared_error(y_test_reg, y_pred_reg)

r2 = r2_score(y_test_reg, y_pred_reg)

print("=" * 60)
print("Linear Regression Performance")
print("=" * 60)

print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"R² Score               : {r2:.3f}")
# Feature Coefficients
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": linear_model.coef_
})

coefficients["Absolute"] = coefficients["Coefficient"].abs()

coefficients = coefficients.sort_values(
    by="Absolute",
    ascending=False
)

print("=" * 60)
print("Feature Coefficients")
print("=" * 60)

print(coefficients)

print("\nTop 3 Most Important Features")

print(coefficients.head(3))

# =====================================================
# Ridge Regression
# =====================================================

ridge_model = Ridge(alpha=1.0)

ridge_model.fit(X_train_reg_scaled, y_train_reg)

print("Ridge Regression model trained successfully.")
# =====================================================
# Ridge Predictions
# =====================================================

ridge_pred = ridge_model.predict(X_test_reg_scaled)
# =====================================================
# Ridge Performance
# =====================================================

ridge_mse = mean_squared_error(y_test_reg, ridge_pred)

ridge_r2 = r2_score(y_test_reg, ridge_pred)

print("=" * 60)
print("Ridge Regression Performance")
print("=" * 60)

print(f"MSE : {ridge_mse:.3f}")
print(f"R²  : {ridge_r2:.3f}")
# =====================================================
# Model Comparison
# =====================================================

comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Ridge Regression"],
    "MSE": [mse, ridge_mse],
    "R² Score": [r2, ridge_r2]
})

print("=" * 60)
print("Model Comparison")
print("=" * 60)

print(comparison)
# =====================================================
# Check Class Distribution
# =====================================================

print("=" * 60)
print("Training Class Distribution")
print("=" * 60)

print(y_train_clf.value_counts())

print("\nPercentage")

print(y_train_clf.value_counts(normalize=True) * 100)
# =====================================================
# Logistic Regression
# =====================================================

log_model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)

log_model.fit(X_train_clf_scaled, y_train_clf)

print("Logistic Regression model trained successfully.")
# =====================================================
# Predictions
# =====================================================

y_pred = log_model.predict(X_test_clf_scaled)

y_prob = log_model.predict_proba(X_test_clf_scaled)[:, 1]
# =====================================================
# Confusion Matrix
# =====================================================

cm = confusion_matrix(y_test_clf, y_pred)

print("=" * 60)
print("Confusion Matrix")
print("=" * 60)

print(cm)
# =====================================================
# Classification Report
# =====================================================

print("=" * 60)
print("Classification Report")
print("=" * 60)

print(classification_report(y_test_clf, y_pred))
# =====================================================
# Metrics
# =====================================================

#accuracy = accuracy_score(y_test_clf, y_pred)

#precision = precision_score(y_test_clf, y_pred)

#recall = recall_score(y_test_clf, y_pred)

#f1 = f1_score(y_test_clf, y_pred)

# =====================================================
# Model Performance
# =====================================================

accuracy = accuracy_score(y_test_clf, y_pred)

baseline_precision = precision_score(y_test_clf, y_pred)

baseline_recall = recall_score(y_test_clf, y_pred)

baseline_f1 = f1_score(y_test_clf, y_pred)

baseline_auc = roc_auc_score(y_test_clf, y_prob)

print("=" * 60)
print("Model Performance")
print("=" * 60)

print(f"Accuracy : {accuracy:.3f}")
print(f"Precision: {baseline_precision:.3f}")
print(f"Recall   : {baseline_recall:.3f}")
print(f"F1 Score : {baseline_f1:.3f}")
print(f"AUC Score: {baseline_auc:.3f}")


print("=" * 60)
print("Model Performance")
print("=" * 60)

print(f"Accuracy : {accuracy:.3f}")
print(f"Precision: {baseline_precision:.3f}")
print(f"Recall   : {baseline_recall:.3f}")
print(f"F1 Score : {baseline_f1:.3f}")
print(f"AUC Score: {baseline_auc:.3f}")
# =====================================================
# ROC Curve
# =====================================================

fpr, tpr, thresholds = roc_curve(y_test_clf, y_prob)

auc = roc_auc_score(y_test_clf, y_prob)

plt.figure(figsize=(7,6))

plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")

plt.plot([0,1], [0,1], linestyle="--")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.grid(True)

plt.savefig("roc_curve.png", dpi=300)
plt.savefig("images/roc_curve.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"AUC Score : {auc:.3f}")
# =====================================================
# Decision Threshold Sensitivity
# =====================================================

thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]

results = []

for t in thresholds:

    pred = (y_prob >= t).astype(int)

    precision = precision_score(y_test_clf, pred)

    recall = recall_score(y_test_clf, pred)

    f1 = f1_score(y_test_clf, pred)

    results.append([t, precision, recall, f1])

threshold_table = pd.DataFrame(
    results,
    columns=[
        "Threshold",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

print("=" * 60)
print("Decision Threshold Comparison")
print("=" * 60)

print(threshold_table)
# =====================================================
# Logistic Regression (Strong Regularization)
# =====================================================

log_model_c001 = LogisticRegression(
    C=0.01,
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)

log_model_c001.fit(X_train_clf_scaled, y_train_clf)

y_pred_c001 = log_model_c001.predict(X_test_clf_scaled)
y_prob_c001 = log_model_c001.predict_proba(X_test_clf_scaled)[:, 1]

precision_c001 = precision_score(y_test_clf, y_pred_c001)
recall_c001 = recall_score(y_test_clf, y_pred_c001)
auc_c001 = roc_auc_score(y_test_clf, y_prob_c001)

print("=" * 60)
print("Strong Regularization (C=0.01)")
print("=" * 60)
print(f"Precision : {precision_c001:.3f}")
print(f"Recall    : {recall_c001:.3f}")
print(f"AUC       : {auc_c001:.3f}")

comparison = pd.DataFrame({
    "Model": ["C = 1.0", "C = 0.01"],
    "Precision": [baseline_precision, precision_c001],
    "Recall": [baseline_recall, recall_c001],
    "AUC": [baseline_auc, auc_c001]
})

print(comparison)

# =====================================================
# Bootstrap Confidence Interval for AUC Difference
# =====================================================

import numpy as np
from sklearn.metrics import roc_auc_score

np.random.seed(42)

auc_differences = []

for _ in range(500):

    indices = np.random.choice(
        len(y_test_clf),
        size=len(y_test_clf),
        replace=True
    )

    y_true = y_test_clf.iloc[indices]

    prob_baseline = y_prob[indices]

    prob_regularized = y_prob_c001[indices]

    auc1 = roc_auc_score(y_true, prob_baseline)
    auc2 = roc_auc_score(y_true, prob_regularized)

    auc_differences.append(auc1 - auc2)

mean_difference = np.mean(auc_differences)

lower_ci = np.percentile(auc_differences, 2.5)

upper_ci = np.percentile(auc_differences, 97.5)

print("=" * 60)
print("Bootstrap Confidence Interval")
print("=" * 60)
print(f"Mean AUC Difference : {mean_difference:.4f}")
print(f"95% CI Lower Bound  : {lower_ci:.4f}")
print(f"95% CI Upper Bound  : {upper_ci:.4f}")

