# ==========================================================
# Big Data Practical Project - Healthcare Dataset
# Cleaning + Visualization + MapReduce + Multiple Algorithms.
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# =========================
# 1) Load Dataset
# =========================

DATASET_URL = "healthcare.csv"

df = pd.read_csv(DATASET_URL)

print("Original Shape:", df.shape)
print(df.head())


# =========================
# 2) Clean Column Names
# =========================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)

print("\nColumns after cleaning:")
print(df.columns)


# =========================
# 3) Fix Duplicate / Wrong Column Names
# =========================

if "name" in df.columns:
    df = df.rename(columns={"name": "patient_name"})

if "name.1" in df.columns:
    df = df.rename(columns={"name.1": "age"})

print("\nColumns after rename:")
print(df.columns)


# =========================
# 4) Understand Dataset
# =========================

print("\nDataset Shape:", df.shape)
print("\nData Types:")
print(df.dtypes)

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

print("\nDuplicates Before Cleaning:", df.duplicated().sum())


# =========================
# 5) Remove Duplicates
# =========================

df = df.drop_duplicates()
print("\nShape after removing duplicates:", df.shape)


# =========================
# 6) Fix Text Inconsistent Values
# =========================

text_columns = df.select_dtypes(include=["object", "string"]).columns

for col in text_columns:
    df[col] = df[col].astype(str).str.strip()

cols_to_lower = [
    "gender",
    "blood_type",
    "medical_condition",
    "admission_type",
    "medication",
    "test_results",
    "insurance"
]

for col in cols_to_lower:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower()


# =========================
# 7) Convert Data Types
# =========================

if "age" in df.columns:
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

if "billing_amount" in df.columns:
    df["billing_amount"] = pd.to_numeric(df["billing_amount"], errors="coerce")

if "room_number" in df.columns:
    df["room_number"] = pd.to_numeric(df["room_number"], errors="coerce")

date_columns = ["date_of_admission", "discharge_date"]

for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")


# =========================
# 8) Feature Engineering
# =========================

if "date_of_admission" in df.columns and "discharge_date" in df.columns:
    df["stay_days"] = (df["discharge_date"] - df["date_of_admission"]).dt.days
    df.loc[df["stay_days"] < 0, "stay_days"] = np.nan


# =========================
# 9) Handle Missing Values
# =========================

for col in df.columns:

    if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == "object":
        df[col] = df[col].replace("nan", np.nan)
        mode_value = df[col].mode(dropna=True)

        if len(mode_value) > 0:
            df[col] = df[col].fillna(mode_value[0])
        else:
            df[col] = df[col].fillna("unknown")

    elif pd.api.types.is_datetime64_any_dtype(df[col]):
        mode_value = df[col].mode(dropna=True)

        if len(mode_value) > 0:
            df[col] = df[col].fillna(mode_value[0])

    elif pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())

    else:
        df[col] = df[col].fillna("unknown")


print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nFinal Shape:", df.shape)
print("\nFinal Data Types:")
print(df.dtypes)

print("\nCleaned Data Preview:")
print(df.head())


# =========================
# 10) Save Clean Dataset
# =========================

df.to_csv("cleaned_healthcare_dataset.csv", index=False)
print("\nCleaned dataset saved successfully.")


# ==========================================================
# Part 3: Hadoop / MapReduce Simulation
# ==========================================================

print("\nMapReduce Simulation: Patients Count by Medical Condition")

mapped_data = []

for condition in df["medical_condition"]:
    mapped_data.append((condition, 1))

reduce_result = {}

for key, value in mapped_data:
    reduce_result[key] = reduce_result.get(key, 0) + value

print(reduce_result)


# ==========================================================
# Part 4: Data Analysis & Visualization
# ==========================================================

# Chart 1: Bar Chart
plt.figure(figsize=(8, 5))
df["medical_condition"].value_counts().plot(kind="bar")
plt.title("Number of Patients by Medical Condition")
plt.xlabel("Medical Condition")
plt.ylabel("Number of Patients")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Chart 2: Pie Chart
plt.figure(figsize=(6, 6))
df["gender"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Gender Distribution")
plt.ylabel("")
plt.tight_layout()
plt.show()


# Chart 3: Line Chart
if "date_of_admission" in df.columns:
    admissions_over_time = df.groupby(df["date_of_admission"].dt.to_period("M")).size()
    admissions_over_time.index = admissions_over_time.index.astype(str)

    plt.figure(figsize=(10, 5))
    admissions_over_time.plot(kind="line", marker="o")
    plt.title("Admissions Over Time")
    plt.xlabel("Month")
    plt.ylabel("Number of Admissions")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Chart 4: Histogram
plt.figure(figsize=(8, 5))
df["age"].plot(kind="hist", bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# Extra Chart: Scatter Plot
plt.figure(figsize=(8, 5))
plt.scatter(df["age"], df["billing_amount"])
plt.title("Age vs Billing Amount")
plt.xlabel("Age")
plt.ylabel("Billing Amount")
plt.tight_layout()
plt.show()


# ==========================================================
# Part 5: Model Training
# Target: Test Results
# ==========================================================

TARGET_COLUMN = "test_results"

model_df = df.copy()

drop_cols = [
    "patient_name",
    "doctor",
    "hospital",
    "date_of_admission",
    "discharge_date"
]

for col in drop_cols:
    if col in model_df.columns:
        model_df = model_df.drop(columns=[col])

# One Hot Encoding
model_df = pd.get_dummies(model_df, drop_first=True)

# بعد get_dummies اسم target بيتغير لو كان categorical
target_cols = [col for col in model_df.columns if col.startswith(TARGET_COLUMN + "_")]

if len(target_cols) > 0:
    # لو test_results اتحول لأكتر من عمود، هنستخدم أول عمود كتصنيف بسيط
    y = model_df[target_cols[0]]
    X = model_df.drop(columns=target_cols)
else:
    y = model_df[TARGET_COLUMN]
    X = model_df.drop(columns=[TARGET_COLUMN])

print("\nTarget Distribution:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================
# Train Random Forest
# =========================

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nRandom Forest Model Completed")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ==========================================================
# Extra: Compare Multiple Algorithms
# ==========================================================

algorithms = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC()
}

results = {}

for name, algorithm in algorithms.items():
    algorithm.fit(X_train, y_train)
    prediction = algorithm.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)
    results[name] = accuracy

    print("\n==============================")
    print("Algorithm:", name)
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(y_test, prediction))


print("\nFinal Algorithms Comparison:")
for name, acc in results.items():
    print(name, ":", acc)


# Chart: Algorithms Accuracy Comparison
plt.figure(figsize=(8, 5))
plt.bar(results.keys(), results.values())
plt.title("Algorithms Accuracy Comparison")
plt.xlabel("Algorithm")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
