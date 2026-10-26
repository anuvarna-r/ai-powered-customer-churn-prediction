import pandas as pd
from pathlib import Path

# ==========================================
# AI-POWERED CUSTOMER CHURN PREDICTION
# Data Cleaning and Preprocessing
# ==========================================

# File paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn.csv"
OUTPUT_PATH = BASE_DIR / "data" / "cleaned_customer_churn.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

print("\n========== DATASET OVERVIEW ==========")
print("Dataset Shape:", df.shape)
print("\nFirst 5 Records:")
print(df.head())

# Check missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# Check duplicate customers
print("\n========== DUPLICATE RECORDS ==========")
print("Duplicate Customer IDs:", df["customerID"].duplicated().sum())

# Remove duplicate customers
df = df.drop_duplicates(subset="customerID").copy()

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Handle missing TotalCharges
missing_charges = df["TotalCharges"].isna().sum()
print("\nMissing TotalCharges:", missing_charges)

# New customers with zero tenure may have no charges yet
new_customers = (
    df["TotalCharges"].isna() &
    (df["tenure"] == 0)
)

df.loc[new_customers, "TotalCharges"] = 0

# Drop any remaining records with missing TotalCharges
df = df.dropna(subset=["TotalCharges"]).copy()

# Create binary target variable
df["Churn_Label"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# Validate target
if df["Churn_Label"].isna().any():
    raise ValueError("Unexpected values found in Churn column.")

# Save cleaned dataset
df.to_csv(OUTPUT_PATH, index=False)

print("\n========== CLEANING COMPLETED ==========")
print("Cleaned Dataset Shape:", df.shape)
print("Customer Churn Distribution:")
print(df["Churn"].value_counts())

print("\nChurn Rate: {:.2f}%".format(
    df["Churn_Label"].mean() * 100
))

print("\nCleaned dataset saved successfully!")
print("File:", OUTPUT_PATH)