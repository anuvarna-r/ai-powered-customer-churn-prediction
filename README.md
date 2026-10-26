# ✦ Customer Intelligence Studio

### AI-Powered Customer Churn Prediction & Explainable Analytics

An end-to-end machine learning project designed to predict customer churn, analyze customer behavior, identify at-risk customers, and provide interpretable predictions using Explainable AI (SHAP).

The project combines predictive modeling, business intelligence, and an interactive Streamlit dashboard to support data-driven customer retention decisions.

## Project Overview

Customer churn is a significant challenge for subscription-based businesses. Identifying customers who may discontinue their services allows organizations to prioritize retention strategies.

**Customer Intelligence Studio** addresses this problem through:

* Customer data cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* Machine learning model development
* Cross-validation and model optimization
* Customer churn probability prediction
* Customer risk segmentation
* Explainable AI using SHAP
* Interactive analytics and visualizations

## Dataset

**Dataset:** IBM Telco Customer Churn

| Attribute           | Details                          |
| ------------------- | -------------------------------- |
| Total Customers     | 7,043                            |
| Original Features   | 21 columns, including the target |
| Retained Customers  | 5,174                            |
| Churned Customers   | 1,869                            |
| Customer Churn Rate | 27% (rounded)                    |
| Target Variable     | Churn                            |

The dataset contains customer demographics, subscription services, contract information, payment methods, tenure, and billing details.

## Technology Stack

**Programming:** Python

**Data Processing:** Pandas, NumPy

**Machine Learning:** Scikit-learn

**Explainable AI:** SHAP

**Visualization:** Plotly, Matplotlib, Seaborn

**Web Application:** Streamlit

**Model Serialization:** Joblib

**Development Tools:** VS Code, Git, GitHub

## Machine Learning Development

Three classification algorithms were evaluated:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting

The modeling workflow includes preprocessing, categorical encoding, feature scaling, stratified train-test splitting, and cross-validation.

### Model Optimization

Logistic Regression was selected for the deployed application using a business-cost-based threshold optimization approach.

**Selected Model:** Logistic Regression

**Optimized Decision Threshold:** 0.37

The lower decision threshold prioritizes identifying customers who may churn, improving recall while increasing false-positive predictions.

### Final Test Performance

| Evaluation Metric | Result |
| ----------------- | -----: |
| Accuracy          |    68% |
| Precision         |    45% |
| Recall            |    90% |
| F1 Score          |    60% |
| ROC-AUC           |    84% |

**Confusion Matrix**

|              | Predicted Stay | Predicted Churn |
| ------------ | -------------: | --------------: |
| Actual Stay  |            628 |             407 |
| Actual Churn |             38 |             336 |

The model identifies approximately 90% of customers who actually churn.

However, its lower precision indicates that some customers predicted to churn would actually remain.

The threshold was selected using illustrative business-cost assumptions and should be reassessed before real-world deployment.

## Streamlit Dashboard

The application contains five interactive modules.

### 1. Overview

Provides an overview of customer retention, churn rates, contract behavior, and key business insights.

### 2. Churn Prediction

Allows users to enter customer information and generate:

* Estimated churn probability
* Predicted customer outcome
* Customer risk category
* Churn probability gauge
* Suggested retention actions

### 3. Customer Analytics

Explores customer behavior through interactive charts and segment-level analysis.

Includes contract analysis, payment behavior, monthly charges, and customer tenure.

### 4. Risk Intelligence

Segments customers into three categories:

* High Risk
* Medium Risk
* Low Risk

Includes risk distribution visualizations, customer-level risk exploration, filtering, and downloadable reports.

Risk scores are generated for historical customers, including records used during training. They should not be interpreted as independent validation results.

### 5. Model Lab

Provides machine learning performance insights, including:

* Cross-validation comparisons
* Selected algorithm
* Decision threshold
* Classification metrics
* Confusion matrix

## Explainable AI — SHAP

The project integrates SHAP to explain individual customer churn predictions.

SHAP identifies customer characteristics that increase or decrease the model's predicted churn score relative to a reference background.

Examples of influential characteristics include:

* Customer tenure
* Contract type
* Payment method
* Internet service
* Monthly and total charges

The dashboard displays these contributions through an interactive horizontal bar chart.

**Important:** SHAP contributions for the Logistic Regression model are expressed in log-odds, not percentage changes. They explain model behavior rather than establish causal relationships.

## Project Structure

```text
ai-powered-customer-churn-prediction/
│
├── data/
│   ├── Telco-Customer-Churn.csv
│   └── cleaned_customer_churn.csv
│
├── models/
│   └── optimized_churn_model.pkl
│
├── reports/
│   └── cross_validation_results.csv
│
├── app.py
├── data_cleaning.py
├── churn_analysis.py
├── train_model.py
├── model_optimization.py
├── explainable_ai.py
├── requirements.txt
├── README.md
└── .gitignore
```

The `reports/` folder may also contain additional exploratory analysis charts and model evaluation outputs.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/anuvarna-r/ai-powered-customer-churn-prediction.git
```

### 2. Navigate to the project

```bash
cd ai-powered-customer-churn-prediction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## Business Applications

Customer Intelligence Studio demonstrates how machine learning can support:

* Customer retention planning
* Churn risk identification
* Customer segmentation
* Predictive analytics
* Model interpretation
* Business intelligence reporting

The project is a portfolio demonstration using historical telecommunications data, not a production-ready customer decision system.

## Future Improvements

Potential extensions include:

* Model probability calibration
* Retention campaign effectiveness analysis
* Cost-sensitive model evaluation using validated business assumptions
* Customer-level retention recommendations
* Production monitoring and model drift detection

## Author

**Anuvarna R.**

BCA Data Science Graduate

Python | Data Analytics | Machine Learning | Artificial Intelligence

**GitHub:** anuvarna-r

---

*Customer Intelligence Studio — Turning customer data into actionable intelligence.*
