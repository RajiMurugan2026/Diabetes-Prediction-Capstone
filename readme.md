# Diabetes Prediction Using Machine Learning and LLM

## Student Information

**Student Name:** Sri Rajarajeshwari

## Project Overview

This project develops an end-to-end diabetes prediction system using machine learning and a Large Language Model (LLM). The project is divided into four parts:

* **Part 1:** Data Cleaning and Preprocessing
* **Part 2:** Regression and Classification
* **Part 3:** Advanced Machine Learning Models
* **Part 4:** LLM-Powered Prediction Explanation

Dataset used: **Pima Indians Diabetes Dataset**

---

# Part 1 – Data Cleaning and Preprocessing

## Objective

Prepare the dataset for machine learning by handling missing values, duplicates, and inconsistent data.

## Steps Performed

* Loaded the dataset using Pandas
* Explored dataset structure
* Checked data types
* Identified missing values
* Replaced invalid zero values
* Filled missing values using median imputation
* Removed duplicate records
* Saved the cleaned dataset as `cleaned_data.csv`

## Output

* Clean dataset ready for model building
* No missing values remain

---

# Part 2 – Regression and Classification

## Objective

Build baseline machine learning models for diabetes prediction.

## Models Implemented

### Linear Regression

Performance evaluated using:

* Mean Squared Error (MSE)
* R² Score

### Logistic Regression

Performance evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC Curve
* AUC Score

## Class Imbalance

Handled class imbalance and compared class distributions before and after processing.

## Evaluation Metrics

* Confusion Matrix
* Classification Report
* ROC Curve
* ROC-AUC

---

# Part 3 – Advanced Machine Learning

## Objective

Improve prediction performance using ensemble learning and hyperparameter tuning.

## Models Implemented

* Decision Tree
* Controlled Decision Tree
* Random Forest
* Gradient Boosting
* Logistic Regression (Cross Validation)

## Decision Tree Comparison

### Baseline Tree

Training Accuracy: **1.000**

Testing Accuracy: **0.610**

The model severely overfits the training data.

### Controlled Tree

Training Accuracy: **0.803**

Testing Accuracy: **0.682**

Adding constraints reduced overfitting and improved generalization.

---

## Gini vs Entropy

### Gini Impurity

Gini Index:

**Gini = 1 − Σ(pi²)**

Measures impurity by calculating the probability of incorrect classification.

### Entropy

Entropy:

**Entropy = − Σ(pi log₂ pi)**

Measures randomness or disorder within a node.

Results:

| Criterion | Test Accuracy |
| --------- | ------------: |
| Gini      |         0.675 |
| Entropy   |         0.708 |

Entropy produced slightly better performance.

---

## Random Forest Results

Training Accuracy: **0.987**

Testing Accuracy: **0.682**

ROC-AUC: **0.762**

### Top Five Important Features

| Feature                  | Importance |
| ------------------------ | ---------: |
| BMI                      |      0.225 |
| Age                      |      0.166 |
| DiabetesPedigreeFunction |      0.159 |
| Insulin                  |      0.145 |
| BloodPressure            |      0.104 |

BMI is the most influential predictor.

---

## Feature Ablation Study

Five least important features removed:

* DiabetesPedigreeFunction
* Insulin
* BloodPressure
* Pregnancies
* SkinThickness

Results:

| Model         | ROC-AUC |
| ------------- | ------: |
| Full Model    |   0.762 |
| Reduced Model |   0.741 |

Removing these variables caused only a small reduction in AUC, showing that some predictors contribute relatively little individually. However, keeping them provides slightly better predictive performance, making the full model preferable for production.

---

## GridSearchCV

Hyperparameter Grid

* n_estimators = 50, 100, 200
* max_depth = 5, 10, None
* min_samples_leaf = 1, 5

Best Parameters

* n_estimators = 200
* max_depth = 5
* min_samples_leaf = 1

Best Cross Validation ROC-AUC

**0.769**

---

## Cross Validation Comparison

| Model               | Mean ROC-AUC |
| ------------------- | -----------: |
| Logistic Regression |        0.768 |
| Decision Tree       |        0.708 |
| Random Forest       |        0.745 |
| Gradient Boosting   |        0.733 |

Logistic Regression achieved the highest cross-validation ROC-AUC, while Random Forest provided strong predictive performance and interpretability.

---

## Manual Learning Curve

Training fractions:

* 20%
* 40%
* 60%
* 80%
* 100%

Training and testing AUC values were calculated for each fraction.

### Interpretation

As the training size increased, testing AUC improved while training AUC became more stable. This indicates that the model benefits from additional training data and is more data-limited than capacity-limited.

---

## Model Serialization

The best model obtained through GridSearchCV was saved as:

`best_model.pkl`

The model was successfully reloaded using Joblib and used for prediction on unseen samples.

---

# Part 4 – LLM-Powered Feature

## Selected Track

**Track C – Model Prediction Explanation Pipeline**

## Objective

Integrate a Large Language Model to explain machine learning predictions in structured JSON format.

## Implementation

* Loaded `best_model.pkl`
* Generated predictions using `.predict()`
* Calculated probabilities using `.predict_proba()`
* Sent predictions to an LLM
* Received structured JSON explanations
* Validated responses using JSON Schema

---

## System Prompt

The LLM was instructed to return only valid JSON containing:

* prediction_label
* confidence_level
* top_reason
* second_reason
* next_step

---

## Temperature Comparison

Both temperature values were tested.

| Temperature | Observation                                  |
| ----------- | -------------------------------------------- |
| 0           | Deterministic, consistent JSON output        |
| 0.7         | More varied wording while preserving meaning |

Temperature 0 was selected because deterministic output is preferred for structured prediction explanations.

---

## PII Guardrail

A regular expression checks for:

* Email addresses
* Phone numbers

If PII is detected, the request is blocked before contacting the LLM.

Example:

| Input                                 | Result  |
| ------------------------------------- | ------- |
| [abc@gmail.com](mailto:abc@gmail.com) | Blocked |
| Patient age is 35                     | Allowed |

---

## JSON Schema Validation

Each LLM response was:

* Parsed using `json.loads()`
* Validated using `jsonschema.validate()`

All three responses passed validation successfully.

---

## Demonstration

Three patient records were processed.

For each record:

* Prediction generated
* Probability calculated
* LLM explanation produced
* JSON validated successfully

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Requests
* Joblib
* JSON Schema
* Python-dotenv
* OpenRouter API

---

# Repository Contents

* DataCleaning.py
* Regression_Classification.py
* AdvancedModeling.py
* LLM_Explanation.py
* cleaned_data.csv
* best_model.pkl
* README.md
* requirements.txt

The `.env` file containing API keys is intentionally excluded from the repository.

---

# Conclusion

This project demonstrates a complete machine learning pipeline for diabetes prediction, from data preprocessing to advanced model optimization and deployment with an LLM-powered explanation system. The final solution combines predictive modeling, hyperparameter tuning, model serialization, structured LLM responses, schema validation, and privacy guardrails, providing an end-to-end, production-oriented workflow.
