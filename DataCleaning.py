import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Display all columns
pd.set_option("display.max_columns", None)

# Plot settings
plt.rcParams["figure.figsize"] = (8,5)

print("Libraries imported successfully.")

file_path = r"C:\Users\user\Documents\Capstone project\diabetes.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully.")

print("="*60)
print("First Five Rows")
print("="*60)

print(df.head())

print("\n")

print("="*60)
print("Data Types")
print("="*60)

print(df.dtypes)

print("\n")

print("="*60)
print("Dataset Shape")
print("="*60)

print(df.shape)

# Count the number of zeros in columns where 0 is considered missing

columns_with_zero_missing = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for col in columns_with_zero_missing:
    zero_count = (df[col] == 0).sum()
    print(f"{col}: {zero_count} zero values")

columns_with_zero_missing = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

df[columns_with_zero_missing] = df[columns_with_zero_missing].replace(0, np.nan)

print("Zeros converted to NaN successfully.")

# Missing value count
missing_count = df.isnull().sum()

# Missing value percentage
missing_percentage = (missing_count / len(df)) * 100

# Create a summary table
missing_table = pd.DataFrame({
    "Missing Count": missing_count,
    "Missing Percentage": missing_percentage
})

print("=" * 60)
print("Missing Value Analysis")
print("=" * 60)
print(missing_table)

high_missing = missing_table[missing_table["Missing Percentage"] > 20]

print("\nColumns with more than 20% missing values:")
print(high_missing)

for col in df.select_dtypes(include='number').columns:

    missing_percent = (df[col].isnull().sum() / len(df)) * 100

    if 0 < missing_percent < 20:
        df[col] = df[col].fillna(df[col].median())

print("\nMissing values filled for columns with less than 20% missing data.")

# Duplicate Detection
duplicate_count = df.duplicated().sum()

print("=" * 60)
print("Duplicate Rows")
print("=" * 60)
print(f"Number of duplicate rows: {duplicate_count}")

# Store null percentage before removing duplicates

null_before = (df.isnull().sum() / len(df)) * 100

# Remove duplicate rows

rows_before = df.shape[0]

df = df.drop_duplicates()

rows_after = df.shape[0]

rows_removed = rows_before - rows_after

print("=" * 60)
print("Duplicate Removal Summary")
print("=" * 60)

print(f"Rows before removal : {rows_before}")
print(f"Rows after removal  : {rows_after}")
print(f"Rows removed        : {rows_removed}")

# Null percentage after removing duplicates

null_after = (df.isnull().sum() / len(df)) * 100

comparison = pd.DataFrame({
    "Before (%)": null_before,
    "After (%)": null_after
})

print("=" * 60)
print("Null Percentage Comparison")
print("=" * 60)

print(comparison)

# Data Types Before Conversion
print("=" * 60)
print("Data Types Before Conversion")
print("=" * 60)

print(df.dtypes)
memory_before = df.memory_usage(deep=True).sum()

print(f"Memory Usage Before Conversion: {memory_before} bytes")
df["Outcome"] = df["Outcome"].astype("category")
print("=" * 60)
print("Data Types After Conversion")
print("=" * 60)

print(df.dtypes)

memory_after = df.memory_usage(deep=True).sum()

print(f"Memory Usage After Conversion: {memory_after} bytes")

print(f"Memory Saved: {memory_before - memory_after} bytes")

memory_after = df.memory_usage(deep=True).sum()

print("=" * 60)
print("Memory Usage After Conversion")
print("=" * 60)

print(f"{memory_after:,} bytes")

print("\nMemory Saved:")

print(f"{memory_before-memory_after:,} bytes")

# Descriptive Statistics
print("=" * 60)
print("Descriptive Statistics")
print("=" * 60)

print(df.describe())

# Skewness of Numeric Columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

skewness = df[numeric_cols].skew()

print("=" * 60)
print("Skewness of Numeric Columns")
print("=" * 60)

print(skewness)
# Highest absolute skewness

highest_skew_col = skewness.abs().idxmax()
highest_skew_value = skewness[highest_skew_col]

print("=" * 60)
print("Most Skewed Column")
print("=" * 60)

print(f"Column : {highest_skew_col}")
print(f"Skewness : {highest_skew_value:.3f}")
skew_table = pd.DataFrame({
    "Skewness": skewness,
    "Absolute Skewness": skewness.abs()
})

skew_table = skew_table.sort_values(
    by="Absolute Skewness",
    ascending=False
)

print(skew_table)

# Task 6: Outlier Detection using IQR
columns = ["Insulin", "Glucose"]

for col in columns:

    print("=" * 60)
    print(f"Outlier Analysis for {col}")
    print("=" * 60)

    # Calculate Quartiles
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    # Calculate IQR
    IQR = Q3 - Q1

    # Calculate Bounds
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)

    # Count Outliers
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    print(f"Q1           : {Q1:.2f}")
    print(f"Q3           : {Q3:.2f}")
    print(f"IQR          : {IQR:.2f}")
    print(f"Lower Bound  : {lower_bound:.2f}")
    print(f"Upper Bound  : {upper_bound:.2f}")
    print(f"Outliers     : {len(outliers)}")

    # Outlier Summary
    summary = []

for col in columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    count = ((df[col] < lower) | (df[col] > upper)).sum()

    summary.append([col, Q1, Q3, IQR, lower, upper, count])

outlier_table = pd.DataFrame(
    summary,
    columns=[
        "Column",
        "Q1",
        "Q3",
        "IQR",
        "Lower Bound",
        "Upper Bound",
        "Number of Outliers"
    ]
)

print(outlier_table)

# Create the images folder if it doesn't exist
os.makedirs("images", exist_ok=True)

#1. Line Plot

plt.figure(figsize=(10,5))

plt.plot(df.index, df["Glucose"], color="blue")

plt.title("Glucose Levels Across Dataset")

plt.xlabel("Row Index")

plt.ylabel("Glucose Level")

plt.grid(True)
plt.savefig("images/line_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# 2. Bar Chart
plt.figure(figsize=(6,5))

df.groupby("Outcome")["BMI"].mean().plot(kind="bar")

plt.title("Average BMI by Diabetes Outcome")

plt.xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")

plt.ylabel("Mean BMI")

plt.xticks(rotation=0)
plt.savefig("images/bar_chart.png", dpi=300, bbox_inches="tight")
plt.show()

# 3. Histogram
plt.figure(figsize=(8,5))

sns.histplot(df["Insulin"], bins=20, kde=True)

plt.title("Distribution of Insulin")

plt.xlabel("Insulin")

plt.ylabel("Frequency")
plt.savefig("images/histogram.png", dpi=300, bbox_inches="tight")
plt.show()

# 4. Scatter Plot
plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x="Glucose",
    y="BMI",
    hue="Outcome"
)

plt.title("Glucose vs BMI")

plt.xlabel("Glucose")

plt.ylabel("BMI")
plt.savefig("images/scatter_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# 5. Box Plot
plt.figure(figsize=(7,5))

sns.boxplot(
    data=df,
    x="Outcome",
    y="Glucose"
)

plt.title("Glucose Distribution by Diabetes Outcome")

plt.xlabel("Outcome")

plt.ylabel("Glucose")
plt.savefig("images/box_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# Task 8: Pearson Correlation Matrix
correlation_matrix = df.corr(numeric_only=True)

print("=" * 60)
print("Pearson Correlation Matrix")
print("=" * 60)

print(correlation_matrix)

# Pearson Correlation Heat Map
plt.figure(figsize=(10,8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    linewidths=0.5,
    fmt=".2f"
)

plt.title("Pearson Correlation Heat Map")
plt.savefig("images/correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

corr = correlation_matrix.abs().copy()

# Set diagonal values to 0
for i in range(len(corr)):
    corr.iat[i, i] = 0

# Stack the matrix and find the maximum correlation
max_pair = corr.stack().idxmax()
max_value = corr.stack().max()

print("=" * 60)
print("Highest Absolute Correlation Pair")
print("=" * 60)

print(f"Variables   : {max_pair[0]} and {max_pair[1]}")
print(f"Correlation : {max_value:.3f}")

# Median Imputation
columns = ["Insulin", "DiabetesPedigreeFunction"]

print("=" * 70)
print("Mean vs Median Comparison")
print("=" * 70)

for col in columns:

    print(f"\nColumn : {col}")

    print(f"Mean   : {df[col].mean():.2f}")

    print(f"Median : {df[col].median():.2f}")

# Median Imputation
# =====================
for col in columns:

    df[col] = df[col].fillna(df[col].median())

print("\nMedian imputation completed.")

# Verify Missing Values
print("=" * 70)
print("Remaining Missing Values")
print("=" * 70)

print(df[columns].isnull().sum())

# Task 9(b): Pearson vs Spearman Correlation
# Select only numeric columns
numeric_df = df.select_dtypes(include=["int64", "float64"])

# Pearson correlation
pearson_corr = numeric_df.corr(method="pearson")

# Spearman correlation
spearman_corr = numeric_df.corr(method="spearman")

print("=" * 60)
print("Pearson Correlation Matrix")
print("=" * 60)
print(pearson_corr)

print("\n")

print("=" * 60)
print("Spearman Correlation Matrix")
print("=" * 60)
print(spearman_corr)

# Absolute Difference

difference = (spearman_corr - pearson_corr).abs()

print("=" * 60)
print("Absolute Difference Matrix")
print("=" * 60)

print(difference)

# Top 3 Largest Differences
# Convert to long format
diff_pairs = difference.stack().reset_index()

diff_pairs.columns = ["Variable 1", "Variable 2", "Difference"]

# Remove self-correlations
diff_pairs = diff_pairs[
    diff_pairs["Variable 1"] != diff_pairs["Variable 2"]
]

# Remove duplicate pairs
diff_pairs["Pair"] = diff_pairs.apply(
    lambda x: tuple(sorted([x["Variable 1"], x["Variable 2"]])),
    axis=1
)

diff_pairs = diff_pairs.drop_duplicates(subset="Pair")

# Sort by difference
top3 = diff_pairs.sort_values(
    by="Difference",
    ascending=False
).head(3)

print("=" * 60)
print("Top 3 Differences")
print("=" * 60)

print(top3[["Variable 1", "Variable 2", "Difference"]])

# Pearson vs Spearman Comparison
for _, row in top3.iterrows():

    v1 = row["Variable 1"]
    v2 = row["Variable 2"]

    pearson = pearson_corr.loc[v1, v2]
    spearman = spearman_corr.loc[v1, v2]

    print("=" * 60)
    print(f"{v1}  <-->  {v2}")

    print(f"Pearson  : {pearson:.3f}")
    print(f"Spearman : {spearman:.3f}")
    print(f"|Difference| : {abs(spearman-pearson):.3f}")

# Task 9(c): Grouped Aggregation
grouped = df.groupby("Outcome", observed=False)["Glucose"].agg(
    ["mean", "std", "count"]
)

print("=" * 60)
print("Grouped Aggregation")
print("=" * 60)

print(grouped)

# Highest Mean and Standard Deviation
highest_mean_group = grouped["mean"].idxmax()
highest_mean = grouped["mean"].max()

highest_std_group = grouped["std"].idxmax()
highest_std = grouped["std"].max()

print("=" * 60)
print("Highest Mean")
print("=" * 60)

print(f"Group : {highest_mean_group}")
print(f"Mean  : {highest_mean:.2f}")

print("\n")

print("=" * 60)
print("Highest Standard Deviation")
print("=" * 60)

print(f"Group : {highest_std_group}")
print(f"Standard Deviation : {highest_std:.2f}")

# Mean Ratio
highest_mean = grouped["mean"].max()
lowest_mean = grouped["mean"].min()

ratio = highest_mean / lowest_mean

print("=" * 60)
print("Mean Ratio")
print("=" * 60)

print(f"Highest Mean : {highest_mean:.2f}")
print(f"Lowest Mean  : {lowest_mean:.2f}")
print(f"Ratio        : {ratio:.2f}")

# Save Cleaned Dataset
df.to_csv("cleaned_data.csv", index=False)

print("=" * 60)
print("cleaned_data.csv has been saved successfully.")
print("=" * 60)



