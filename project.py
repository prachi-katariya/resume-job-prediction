# ============================================================
# 1. DATASET COLLECTION
# Resume-Based Job Selection Prediction
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Load Dataset
file_path = r"E:\project1\ml_resume_dataset_4500.csv"
df = pd.read_csv(file_path)

print(df.head())
print(df.shape)

print(df.info())
print(df.describe())

# ============================================================
# 2. UNDERSTANDING DATA
# ============================================================


print("\n========== UNDERSTANDING DATA ==========")

# Dataset Information
print("\nDataset Information:")
print(df.info())

# Numerical Columns
print("\nNumerical Columns:")
print(df.select_dtypes(include=["int64", "float64"]).columns.tolist())

# Categorical Columns
print("\nCategorical Columns:")
print(df.select_dtypes(include=["object", "bool"]).columns.tolist())

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate Values
print("\nDuplicate Records:")
print(df.duplicated().sum())

# Dataset Summary
print("\nDataset Summary:")
print(df.describe(include="all"))

# ============================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

print("\n========== EXPLORATORY DATA ANALYSIS ==========")


# ------------------------------------------------------------
# 3.1 Histogram - Years of Experience
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["years_experience"],
    bins=10
)

plt.title("Distribution of Years of Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Number of Candidates")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 3.2 Box Plot - Years of Experience
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.boxplot(
    df["years_experience"].dropna()
)

plt.title("Box Plot of Years of Experience")
plt.ylabel("Years of Experience")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 3.3 Pie Chart - Job Selection
# ------------------------------------------------------------

label_counts = df["label"].value_counts()

plt.figure(figsize=(6, 6))

plt.pie(
    label_counts,
    labels=["Not Selected", "Selected"],
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Job Selection Distribution")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 3.4 Bar Chart - Highest Degree
# ------------------------------------------------------------

degree_counts = df["highest_degree"].value_counts()

plt.figure(figsize=(8, 5))

degree_counts.plot(
    kind="bar"
)

plt.title("Candidates by Highest Degree")
plt.xlabel("Highest Degree")
plt.ylabel("Number of Candidates")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 3.5 Correlation Matrix
# ------------------------------------------------------------

numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
)

correlation = numeric_columns.corr()

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Matrix")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# EDA Completed
# ------------------------------------------------------------

print("\nEDA Completed Successfully!")

# ============================================================
# 4. HANDLE MISSING VALUES & OUTLIERS
# ============================================================

print("\n========== PART 4: DATA HANDLING ==========")


# ------------------------------------------------------------
# 4.1 Check Missing Values
# ------------------------------------------------------------

print("\nMissing Values:")

missing_values = df.isnull().sum()

print(missing_values)

if missing_values.sum() == 0:
    print("\nNo missing values found in the dataset.")
else:
    print("\nMissing values are present in the dataset.")


# ------------------------------------------------------------
# 4.2 Outlier Detection using IQR
# ------------------------------------------------------------

print("\n========== OUTLIER DETECTION ==========")

# Numerical column
column = "years_experience"

Q1 = df[column].quantile(0.25)
Q3 = df[column].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

print("\nQ1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)


# Find outliers
outliers = df[
    (df[column] < lower_limit) |
    (df[column] > upper_limit)
]

print("\nNumber of Outliers BEFORE Handling:")
print(len(outliers))


# ------------------------------------------------------------
# 4.3 Before Handling
# ------------------------------------------------------------

before_count = len(outliers)

print("\nBefore Handling:")
print("Total Records:", len(df))
print("Outliers:", before_count)


# ------------------------------------------------------------
# 4.4 Handle Outliers using Clipping
# ------------------------------------------------------------

df_clean = df.copy()

df_clean[column] = df_clean[column].clip(
    lower=lower_limit,
    upper=upper_limit
)


# ------------------------------------------------------------
# 4.5 Check Outliers AFTER Handling
# ------------------------------------------------------------

outliers_after = df_clean[
    (df_clean[column] < lower_limit) |
    (df_clean[column] > upper_limit)
]

after_count = len(outliers_after)

print("\nAfter Handling:")
print("Total Records:", len(df_clean))
print("Outliers:", after_count)


# ------------------------------------------------------------
# 4.6 Before vs After Comparison
# ------------------------------------------------------------

comparison = pd.DataFrame({
    "Before": [before_count],
    "After": [after_count]
})

print("\n========== BEFORE VS AFTER COMPARISON ==========")

print(comparison)


# ------------------------------------------------------------
# 4.7 Box Plot After Handling
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.boxplot(
    df_clean[column].dropna()
)

plt.title("Box Plot After Outlier Handling")
plt.ylabel("Years of Experience")

plt.tight_layout()
plt.show()


print("\nPart 4 Completed Successfully!")

# ============================================================
# 5. AUTOMATED EDA
# Using Sweetviz
# ============================================================

print("\n========== PART 5: AUTOMATED EDA ==========")

import sweetviz as sv
import os

# Create Reports folder
reports_folder = r"E:\project1\Reports"
os.makedirs(reports_folder, exist_ok=True)

# Create automated EDA report
report = sv.analyze(df_clean)

# Save HTML report
report_path = os.path.join(reports_folder, "EDA_Report.html")

report.show_html(
    report_path,
    open_browser=False
)

print("\nAutomated EDA Report Generated Successfully!")
print("Report saved as:", report_path)

# ============================================================
# 6. FEATURE SELECTION & TRAIN-TEST SPLIT
# ============================================================

print("\n========== PART 6: FEATURE SELECTION ==========")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# ------------------------------------------------------------
# 6.1 Create Working Dataset
# ------------------------------------------------------------

df_model = df_clean.copy()


# ------------------------------------------------------------
# 6.2 Create Combined Resume Text
# ------------------------------------------------------------

df_model["combined_text"] = (
    df_model["skills"].fillna("").astype(str) + " " +
    df_model["current_title"].fillna("").astype(str) + " " +
    df_model["raw_text"].fillna("").astype(str)
)


# ------------------------------------------------------------
# 6.3 Define Independent and Dependent Variables
# ------------------------------------------------------------

X = df_model["combined_text"]
y = df_model["label"]

print("\nDependent Variable (Target):")
print("label")

print("\nIndependent Variable:")
print("skills + current_title + raw_text")


# ------------------------------------------------------------
# 6.4 Train-Test Split
# IMPORTANT: Split BEFORE TF-IDF
# ------------------------------------------------------------

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n========== TRAIN-TEST SPLIT ==========")

print("Total Records:", len(df_model))
print("Training Records:", len(X_train_text))
print("Testing Records:", len(X_test_text))


# ------------------------------------------------------------
# 6.5 TF-IDF Vectorization
# IMPORTANT:
# fit_transform ONLY on training data
# transform ONLY on testing data
# ------------------------------------------------------------

vectorizer = TfidfVectorizer(
    max_features=3000,
    stop_words="english"
)

X_train = vectorizer.fit_transform(X_train_text)

X_test = vectorizer.transform(X_test_text)


# ------------------------------------------------------------
# 6.6 Display Feature Information
# ------------------------------------------------------------

print("\n========== TF-IDF FEATURES ==========")

print("Training Feature Shape:")
print(X_train.shape)

print("\nTesting Feature Shape:")
print(X_test.shape)

print("\nNumber of TF-IDF Features:")
print(len(vectorizer.get_feature_names_out()))


# ------------------------------------------------------------
# 6.7 Final Information
# ------------------------------------------------------------

print("\n========== FEATURE SELECTION SUMMARY ==========")

print("Target Variable: label")
print("Text Features: skills + current_title + raw_text")
print("TF-IDF: Applied")
print("Data Leakage: Avoided")
print("Test Size: 20%")
print("Random State: 42")


print("\nPart 6 Completed Successfully!")
# ============================================================
# 7. BUILD MACHINE LEARNING MODEL
# Logistic Regression
# ============================================================

print("\n========== PART 7: LOGISTIC REGRESSION ==========")

from sklearn.linear_model import LogisticRegression

# Create Logistic Regression model
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

print("\nLogistic Regression Model Trained Successfully!")

# Prediction
y_pred = model.predict(X_test)

print("\nPrediction Completed!")

# ============================================================
# 9. IMPROVE MODEL
# Cross Validation + Regularization
# ============================================================

print("\n========== PART 9: MODEL IMPROVEMENT ==========")

from sklearn.model_selection import cross_val_score

# ------------------------------------------------------------
# 9.1 Original Model Performance
# ------------------------------------------------------------

original_pred = model.predict(X_test)

original_accuracy = accuracy_score(
    y_test, original_pred
)

original_precision = precision_score(
    y_test, original_pred, zero_division=0
)

original_recall = recall_score(
    y_test, original_pred, zero_division=0
)

original_f1 = f1_score(
    y_test, original_pred, zero_division=0
)


# ------------------------------------------------------------
# 9.2 Cross Validation
# ------------------------------------------------------------

cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("\n5-Fold Cross Validation Scores:")
print(cv_scores)

print(
    "\nMean Cross Validation Accuracy:",
    round(cv_scores.mean(), 4)
)


# ------------------------------------------------------------
# 9.3 Improved Logistic Regression
# ------------------------------------------------------------

improved_model = LogisticRegression(
    C=0.5,
    max_iter=1000,
    random_state=42
)

improved_model.fit(
    X_train,
    y_train
)


# ------------------------------------------------------------
# 9.4 Prediction
# ------------------------------------------------------------

y_pred_improved = improved_model.predict(X_test)


# ------------------------------------------------------------
# 9.5 Improved Model Evaluation
# ------------------------------------------------------------

improved_accuracy = accuracy_score(
    y_test,
    y_pred_improved
)

improved_precision = precision_score(
    y_test,
    y_pred_improved,
    zero_division=0
)

improved_recall = recall_score(
    y_test,
    y_pred_improved,
    zero_division=0
)

improved_f1 = f1_score(
    y_test,
    y_pred_improved,
    zero_division=0
)


# ------------------------------------------------------------
# 9.6 Display Results
# ------------------------------------------------------------

print("\n========== ORIGINAL MODEL ==========")

print("Accuracy  :", round(original_accuracy, 4))
print("Precision :", round(original_precision, 4))
print("Recall    :", round(original_recall, 4))
print("F1 Score  :", round(original_f1, 4))


print("\n========== IMPROVED MODEL ==========")

print("Accuracy  :", round(improved_accuracy, 4))
print("Precision :", round(improved_precision, 4))
print("Recall    :", round(improved_recall, 4))
print("F1 Score  :", round(improved_f1, 4))


# ------------------------------------------------------------
# 9.7 Before vs After Comparison
# ------------------------------------------------------------

comparison = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],
    "Original Model": [
        original_accuracy,
        original_precision,
        original_recall,
        original_f1
    ],
    "Improved Model": [
        improved_accuracy,
        improved_precision,
        improved_recall,
        improved_f1
    ]
})

print("\n========== MODEL COMPARISON ==========")
print(comparison)

print("\nPart 9 Completed Successfully!")

# ============================================================
# 10. FINAL RESULTS & CONCLUSION
# ============================================================

print("\n========== PART 10: FINAL RESULTS ==========")


# ------------------------------------------------------------
# 10.1 Final Model Results
# ------------------------------------------------------------

print("\nFinal Model Performance:")
print("-------------------------")

print("Accuracy  :", round(improved_accuracy, 4))
print("Precision :", round(improved_precision, 4))
print("Recall    :", round(improved_recall, 4))
print("F1 Score  :", round(improved_f1, 4))


# ------------------------------------------------------------
# 10.2 Convert Scores to Percentage
# ------------------------------------------------------------

print("\nPerformance in Percentage:")

print("Accuracy  :", round(improved_accuracy * 100, 2), "%")
print("Precision :", round(improved_precision * 100, 2), "%")
print("Recall    :", round(improved_recall * 100, 2), "%")
print("F1 Score  :", round(improved_f1 * 100, 2), "%")


# ------------------------------------------------------------
# 10.3 Test with a New Resume
# ------------------------------------------------------------

new_resume = """
Python machine learning data science SQL pandas numpy
scikit-learn data analysis communication
"""

# Convert new resume into TF-IDF
new_resume_tfidf = vectorizer.transform([new_resume])

# Predict
prediction = improved_model.predict(new_resume_tfidf)[0]

print("\n========== NEW RESUME PREDICTION ==========")

if prediction == 1:
    print("Prediction: JOB SELECTED / SUITABLE")
else:
    print("Prediction: NOT SELECTED / NOT SUITABLE")


# ------------------------------------------------------------
# 10.4 Final Conclusion
# ------------------------------------------------------------

print("\n========== CONCLUSION ==========")

print("""
The Resume-Based Job Selection Prediction project was
successfully developed using Machine Learning.

The dataset was analyzed, cleaned, explored and transformed
using TF-IDF feature extraction.

Logistic Regression was used as the classification model.

The model was evaluated using Accuracy, Precision, Recall
and F1 Score.

Cross Validation and Regularization were also applied to
improve the model performance.

The trained model can predict whether a resume is suitable
for job selection based on the information present in the
resume.
""")


print("\nProject Completed Successfully!")