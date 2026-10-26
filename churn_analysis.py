import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==========================================
# AI-POWERED CUSTOMER CHURN PREDICTION
# Exploratory Data Analysis (EDA)
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "cleaned_customer_churn.csv"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)

# Load cleaned dataset
df = pd.read_csv(DATA_PATH)

# Set visualization theme
sns.set_theme(style="whitegrid", palette="Set2")

print("\n========== CUSTOMER CHURN ANALYSIS ==========")
print("Total Customers:", len(df))
print("Churn Rate: {:.2f}%".format(df["Churn_Label"].mean() * 100))


# ==========================================
# 1. CUSTOMER CHURN DISTRIBUTION
# ==========================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Churn",
    order=["No", "Yes"],
    hue="Churn",
    palette=["#76C7C0", "#FF8C94"],
    legend=False
)

plt.title("Customer Churn Distribution", fontsize=15)
plt.xlabel("Churn Status")
plt.ylabel("Number of Customers")
plt.tight_layout()

plt.savefig(REPORT_DIR / "churn_distribution.png", dpi=300)
plt.close()


# ==========================================
# 2. CHURN RATE BY CONTRACT TYPE
# ==========================================

contract_churn = (
    df.groupby("Contract")["Churn_Label"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

plt.figure(figsize=(9, 5))

sns.barplot(
    x=contract_churn.index,
    y=contract_churn.values,
    hue=contract_churn.index,
    palette="viridis",
    legend=False
)

plt.title("Customer Churn Rate by Contract Type", fontsize=15)
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")
plt.tight_layout()

plt.savefig(REPORT_DIR / "contract_churn.png", dpi=300)
plt.close()


# ==========================================
# 3. CHURN RATE BY PAYMENT METHOD
# ==========================================

payment_churn = (
    df.groupby("PaymentMethod")["Churn_Label"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

plt.figure(figsize=(11, 6))

sns.barplot(
    x=payment_churn.values,
    y=payment_churn.index,
    hue=payment_churn.index,
    palette="coolwarm",
    legend=False
)

plt.title("Customer Churn Rate by Payment Method", fontsize=15)
plt.xlabel("Churn Rate (%)")
plt.ylabel("Payment Method")
plt.tight_layout()

plt.savefig(REPORT_DIR / "payment_churn.png", dpi=300)
plt.close()


# ==========================================
# 4. MONTHLY CHARGES VS CHURN
# ==========================================

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges",
    hue="Churn",
    palette="Set2",
    legend=False
)

plt.title("Monthly Charges vs Customer Churn", fontsize=15)
plt.xlabel("Churn Status")
plt.ylabel("Monthly Charges")
plt.tight_layout()

plt.savefig(REPORT_DIR / "monthly_charges_churn.png", dpi=300)
plt.close()


# ==========================================
# 5. CUSTOMER TENURE ANALYSIS
# ==========================================

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="tenure",
    hue="Churn",
    bins=30,
    kde=True,
    element="step",
    palette={"No": "#76C7C0", "Yes": "#FF8C94"}
)

plt.title("Customer Tenure Distribution by Churn", fontsize=15)
plt.xlabel("Tenure (Months)")
plt.ylabel("Number of Customers")
plt.tight_layout()

plt.savefig(REPORT_DIR / "tenure_analysis.png", dpi=300)
plt.close()


# ==========================================
# 6. NUMERICAL FEATURE CORRELATION
# ==========================================

numerical_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Churn_Label"
]

correlation = df[numerical_features].corr()

plt.figure(figsize=(9, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Customer Churn Feature Correlation", fontsize=15)
plt.tight_layout()

plt.savefig(REPORT_DIR / "correlation_heatmap.png", dpi=300)
plt.close()


# ==========================================
# BUSINESS INSIGHTS
# ==========================================

print("\n========== CHURN BY CONTRACT ==========")
print(contract_churn.round(2).to_string())

print("\n========== CHURN BY PAYMENT METHOD ==========")
print(payment_churn.round(2).to_string())

print("\n========== CUSTOMER TENURE ==========")
print(df.groupby("Churn")["tenure"].mean().round(2))

print("\n========== MONTHLY CHARGES ==========")
print(df.groupby("Churn")["MonthlyCharges"].mean().round(2))

print("\n========== EDA COMPLETED ==========")
print("Six visualizations saved in:", REPORT_DIR)