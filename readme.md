# Diabetes Prediction Capstone Project – Part 1: Data Acquisition, Cleaning and Exploratory Data Analysis

## Project Overview

This project is the first phase of a Diabetes Prediction Capstone Project. The objective is to prepare a real-world healthcare dataset for machine learning by performing data acquisition, data cleaning, exploratory data analysis (EDA), visualization, and statistical analysis. The cleaned dataset produced in this phase will be used in Part 2 to build predictive machine learning models for diabetes prediction.

---

## Problem Statement

Healthcare organizations collect large amounts of patient data, but raw datasets often contain missing values, duplicate records, inconsistent data types, skewed distributions, and outliers. These issues can reduce the performance of machine learning models.

The objective of this project is to clean and analyze the diabetes dataset, understand the relationships between variables, and prepare a high-quality dataset for predictive modeling.

---

## Dataset

**Dataset:** Pima Indians Diabetes Dataset

The dataset contains medical measurements collected from female patients and is commonly used for diabetes prediction research.

### Features

* Pregnancies
* Glucose
* BloodPressure
* SkinThickness
* Insulin
* BMI
* DiabetesPedigreeFunction
* Age
* Outcome (Target Variable)

If the dataset is not included in this repository, it can be downloaded from:

https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Jupyter Notebook / Python Script

---

## Data Cleaning

The following preprocessing steps were completed:

* Loaded the dataset into a Pandas DataFrame.
* Displayed the first five records.
* Inspected data types.
* Checked the dataset dimensions.
* Converted medically impossible zero values to missing values (NaN) for selected columns.
* Calculated missing value counts and percentages.
* Filled missing values using the median where appropriate.
* Detected and removed duplicate rows.
* Converted the Outcome column to the Category data type.
* Compared memory usage before and after data type conversion.

---

## Descriptive Statistics

Descriptive statistics were generated using `describe()` to summarize each numeric feature.

The following statistics were analyzed:

* Count
* Mean
* Standard Deviation
* Minimum
* Maximum
* Quartiles

---

## Skewness Analysis

Skewness was calculated for every numeric feature.

### Most Skewed Features

| Feature                  | Skewness |
| ------------------------ | -------: |
| Insulin                  |    2.166 |
| DiabetesPedigreeFunction |    1.920 |
| Age                      |    1.130 |

The Insulin feature exhibited the highest positive skewness, indicating that a small number of patients have unusually high insulin values. Because the mean is sensitive to extreme values, the median was selected for imputing missing values in skewed variables.

---

## Outlier Detection

Outliers were detected using the Interquartile Range (IQR) method.

The following steps were performed:

* Calculated Q1 and Q3
* Computed the IQR
* Determined lower and upper bounds
* Counted observations outside these bounds

Outliers were documented but retained because they may represent valid clinical measurements. They will be considered during model development in Part 2.

---

## Visualizations

The following visualizations were created:

### 1. Line Plot

Shows variation in glucose values across the dataset.

### 2. Bar Chart

Compares the average BMI for diabetic and non-diabetic patients.

### 3. Histogram

Displays the distribution of Insulin values and confirms a positively skewed distribution.

### 4. Scatter Plot

Illustrates the relationship between Glucose and BMI.

### 5. Box Plot

Compares Glucose distributions between diabetic and non-diabetic patients, highlighting differences in median and variability.

### 6. Correlation Heat Map

Visualizes Pearson correlations among all numeric variables.

The strongest Pearson correlation was observed between **SkinThickness** and **BMI** with a correlation coefficient of **0.648**, indicating a moderately strong positive linear relationship. This relationship does not imply causation because both variables may be influenced by underlying factors such as body composition and overall health.

---

## Pearson vs. Spearman Correlation

Both Pearson and Spearman correlation matrices were computed and compared.

The largest differences were observed for:

| Variable Pair         | Pearson | Spearman |
| --------------------- | ------: | -------: |
| BMI – Age             |   0.026 |    0.121 |
| Insulin – Age         |   0.097 |    0.188 |
| Pregnancies – Insulin |   0.025 |    0.096 |

These results indicate monotonic relationships that are not strongly linear. Pearson correlation will be used primarily for feature-selection guidance in the next phase, while Spearman correlation provides additional insight for skewed variables.

---

## Grouped Aggregation

Grouped aggregation was performed using the Outcome category to calculate the mean, standard deviation, and count of Glucose values.

This analysis was used to compare diabetic and non-diabetic groups and evaluate whether Outcome provides predictive information for Glucose levels.

---

## Output

The cleaned dataset was saved as:

`cleaned_data.csv`

This file will be used for machine learning in Part 2.

---

## Repository Structure

```text
Diabetes-Prediction-Capstone/
│
├── DataCleaning.py
├── diabetes.csv
├── cleaned_data.csv
├── README.md
├── requirements.txt
└── images/
    ├── line_plot.png
    ├── bar_chart.png
    ├── histogram.png
    ├── scatter_plot.png
    ├── box_plot.png
    └── correlation_heatmap.png
```

---

## How to Run

1. Install the required libraries:

```bash
pip install -r requirements.txt
```

2. Place `diabetes.csv` in the project folder.

3. Run the Python script:

```bash
python DataCleaning.py
```

4. The script will:

* Clean the dataset
* Generate all required visualizations
* Save `cleaned_data.csv`

---

## Future Work

Part 2 of this capstone project will include:

* Feature selection
* Data scaling
* Train-test split
* Machine learning model development
* Model evaluation
* Hyperparameter tuning
* Performance comparison
* Final model selection

---

## Author

**Sri Rajarajeshwari**

Capstone Project – Data Science
