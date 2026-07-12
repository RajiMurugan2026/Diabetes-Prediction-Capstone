import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
print("="*60)
print("Loading Dataset")
print("="*60)

df = pd.read_csv("cleaned_data.csv")

# Fill remaining missing values
df["SkinThickness"] = df["SkinThickness"].fillna(
    df["SkinThickness"].median()
)

print("\nRemaining Missing Values")
print(df.isnull().sum())

print(df.head())
X = df.drop(columns=["Outcome", "Glucose"])

y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("="*60)
print("NaN in X_train BEFORE scaling")
print("="*60)

print(X_train.isnull().sum())

print("\nRows with NaN:")
print(X_train[X_train.isnull().any(axis=1)].head())
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
tree = DecisionTreeClassifier(random_state=42)

tree.fit(X_train_scaled, y_train)

print("Decision Tree trained successfully.")
train_pred = tree.predict(X_train_scaled)

test_pred = tree.predict(X_test_scaled)
train_acc = accuracy_score(y_train, train_pred)

test_acc = accuracy_score(y_test, test_pred)

print("="*60)
print("Decision Tree (Baseline)")
print("="*60)

print(f"Training Accuracy : {train_acc:.3f}")
print(f"Testing Accuracy  : {test_acc:.3f}")

print("="*60)
print("Controlled Decision Tree")
print("="*60)

controlled_tree = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=20,
    random_state=42
)

controlled_tree.fit(X_train_scaled, y_train)

train_pred2 = controlled_tree.predict(X_train_scaled)
test_pred2 = controlled_tree.predict(X_test_scaled)

train_acc2 = accuracy_score(y_train, train_pred2)
test_acc2 = accuracy_score(y_test, test_pred2)

print(f"Training Accuracy : {train_acc2:.3f}")
print(f"Testing Accuracy  : {test_acc2:.3f}")
 
print("="*60)
print("Gini vs Entropy")
print("="*60)

# Gini Decision Tree
gini_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

gini_tree.fit(X_train_scaled, y_train)

gini_pred = gini_tree.predict(X_test_scaled)

gini_acc = accuracy_score(y_test, gini_pred)

# Entropy Decision Tree
entropy_tree = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)

entropy_tree.fit(X_train_scaled, y_train)

entropy_pred = entropy_tree.predict(X_test_scaled)

entropy_acc = accuracy_score(y_test, entropy_pred)

print(f"Gini Test Accuracy    : {gini_acc:.3f}")
print(f"Entropy Test Accuracy : {entropy_acc:.3f}")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

print("="*60)
print("Random Forest")
print("="*60)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

rf.fit(X_train_scaled, y_train)

# Predictions
train_pred_rf = rf.predict(X_train_scaled)
test_pred_rf = rf.predict(X_test_scaled)

# Probabilities for ROC-AUC
y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]

# Accuracy
train_acc_rf = accuracy_score(y_train, train_pred_rf)
test_acc_rf = accuracy_score(y_test, test_pred_rf)

# ROC-AUC
auc_rf = roc_auc_score(y_test, y_prob_rf)

print(f"Training Accuracy : {train_acc_rf:.3f}")
print(f"Testing Accuracy  : {test_acc_rf:.3f}")
print(f"ROC-AUC           : {auc_rf:.3f}")

# Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 5 Important Features")
print("="*60)
print(importance.head())

print("=" * 60)
print("Feature Ablation Study")
print("=" * 60)

# Full model AUC
full_auc = auc_rf

# Five least important features
least_features = importance.tail(5)["Feature"].tolist()

print("Lowest 5 Important Features")
print(least_features)

# Remove the least important features
X_reduced = X.drop(columns=least_features)

# Split again
X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(
    X_reduced,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Scale reduced dataset
scaler_red = StandardScaler()

X_train_red_scaled = scaler_red.fit_transform(X_train_red)
X_test_red_scaled = scaler_red.transform(X_test_red)

# Train Random Forest again
rf_reduced = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

rf_reduced.fit(X_train_red_scaled, y_train_red)

# Predict probabilities
y_prob_red = rf_reduced.predict_proba(X_test_red_scaled)[:, 1]

# Reduced AUC
reduced_auc = roc_auc_score(y_test_red, y_prob_red)

print(f"Full Model AUC    : {full_auc:.3f}")
print(f"Reduced Model AUC : {reduced_auc:.3f}")
from sklearn.ensemble import GradientBoostingClassifier

print("="*60)
print("Checking Missing Values")
print("="*60)


from sklearn.ensemble import GradientBoostingClassifier

print("=" * 60)
print("Gradient Boosting")
print("=" * 60)

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

print("=" * 60)
print("5-Fold Cross Validation")
print("=" * 60)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("=" * 60)
print("Grid Search CV")
print("=" * 60)

pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    RandomForestClassifier(random_state=42)
)

param_grid = {
    "randomforestclassifier__n_estimators": [50, 100, 200],
    "randomforestclassifier__max_depth": [5, 10, None],
    "randomforestclassifier__min_samples_leaf": [1, 5]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=1
)

grid.fit(X_train, y_train)

print("Best Parameters")
print(grid.best_params_)

print("\nBest Cross Validation AUC")
print(round(grid.best_score_, 3))

total_models = (
    len(param_grid["randomforestclassifier__n_estimators"])
    * len(param_grid["randomforestclassifier__max_depth"])
    * len(param_grid["randomforestclassifier__min_samples_leaf"])
)

print(f"\nTotal Model Configurations : {total_models}")
print(f"Total Fits Performed       : {total_models * 5}")

# ===========================================
# Save Best Model
# ===========================================

best_pipeline = grid.best_estimator_

joblib.dump(best_pipeline, "best_model.pkl")

print("\nModel saved successfully as best_model.pkl")


# Logistic Regression
log_model = LogisticRegression(max_iter=1000, random_state=42)

# Controlled Decision Tree
tree_cv = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=20,
    random_state=42
)

# Random Forest
rf_cv = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

# Gradient Boosting
gb_cv = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

models = {
    "Logistic Regression": log_model,
    "Decision Tree": tree_cv,
    "Random Forest": rf_cv,
    "Gradient Boosting": gb_cv
}

results = []

for name, model in models.items():

    scores = cross_val_score(
        model,
        X_train_scaled,
        y_train,
        cv=cv,
        scoring="roc_auc"
    )

    results.append([
        name,
        scores.mean(),
        scores.std()
    ])

cv_results = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Mean AUC",
        "Std AUC"
    ]
)

print(cv_results)

gb.fit(X_train_scaled, y_train)

train_pred_gb = gb.predict(X_train_scaled)
test_pred_gb = gb.predict(X_test_scaled)

y_prob_gb = gb.predict_proba(X_test_scaled)[:, 1]

train_acc_gb = accuracy_score(y_train, train_pred_gb)
test_acc_gb = accuracy_score(y_test, test_pred_gb)
auc_gb = roc_auc_score(y_test, y_prob_gb)

print(f"Training Accuracy : {train_acc_gb:.3f}")
print(f"Testing Accuracy  : {test_acc_gb:.3f}")
print(f"ROC-AUC           : {auc_gb:.3f}")

print("=" * 60)
print("Manual Learning Curve")
print("=" * 60)

fractions = [0.2, 0.4, 0.6, 0.8, 1.0]

print(f"{'Train Size':<12}{'Train AUC':<15}{'Test AUC':<15}")

for frac in fractions:

    # Select a fraction of the training set
    if frac < 1.0:
        X_sub, _, y_sub, _ = train_test_split(
            X_train,
            y_train,
            train_size=frac,
            stratify=y_train,
            random_state=42
        )
    else:
        X_sub = X_train.copy()
        y_sub = y_train.copy()

    # Train the best pipeline found by GridSearchCV
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=1,
            random_state=42
        )
    )

    model.fit(X_sub, y_sub)

    train_prob = model.predict_proba(X_sub)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_sub, train_prob)
    test_auc = roc_auc_score(y_test, test_prob)

    print(f"{int(frac*100):<12}{train_auc:<15.3f}{test_auc:<15.3f}")

# ===========================================
# Reload Saved Model
# ===========================================

loaded_model = joblib.load("best_model.pkl")

sample = pd.DataFrame([
    [2,70,35,100,30.5,0.45,28],
    [8,80,32,150,36.1,0.75,52]
], columns=X.columns)

prediction = loaded_model.predict(sample)

print("\nPredictions:")
print(prediction)

