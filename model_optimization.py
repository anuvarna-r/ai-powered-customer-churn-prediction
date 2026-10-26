import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    cross_val_predict,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# ==========================================
# AI-POWERED CUSTOMER CHURN PREDICTION
# Model Optimization and Cross-Validation
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "cleaned_customer_churn.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
COST_FALSE_NEGATIVE = 5
COST_FALSE_POSITIVE = 1

# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv(DATA_PATH)

X = df.drop(
    columns=["customerID", "Churn", "Churn_Label"]
)
y = df["Churn_Label"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\n========== DATA LOADED ==========")
print("Training customers:", len(X_train))
print("Testing customers:", len(X_test))

# ==========================================
# 2. PREPROCESSING
# ==========================================

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()


def create_pipeline(model):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "numerical",
                StandardScaler(),
                numerical_features,
            ),
        ]
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])


# ==========================================
# 3. DEFINE MODELS
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE,
    ),
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

# ==========================================
# 4. CROSS-VALIDATION
# ==========================================

print("\n========== CROSS-VALIDATION ==========")

cv_results = []
threshold_results = []
trained_models = {}

thresholds = np.round(
    np.arange(0.10, 0.81, 0.01), 2
)

for name, estimator in models.items():

    print(f"\nEvaluating: {name}")

    pipeline = create_pipeline(estimator)

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "roc_auc": "roc_auc",
            "average_precision": "average_precision",
        },
        n_jobs=1,
    )

    mean_auc = scores["test_roc_auc"].mean()
    std_auc = scores["test_roc_auc"].std()
    mean_ap = scores["test_average_precision"].mean()

    print(f"Mean ROC-AUC: {mean_auc:.4f}")
    print(f"ROC-AUC Std: {std_auc:.4f}")
    print(f"Mean Average Precision: {mean_ap:.4f}")

    cv_results.append({
        "Model": name,
        "CV_ROC_AUC": mean_auc,
        "CV_ROC_AUC_STD": std_auc,
        "CV_Average_Precision": mean_ap,
    })

    # Out-of-fold predictions use only training data.
    oof_probabilities = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    for threshold in thresholds:

        predictions = (
            oof_probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_train,
            predictions,
            labels=[0, 1],
        ).ravel()

        cost = (
            COST_FALSE_NEGATIVE * fn
            + COST_FALSE_POSITIVE * fp
        )

        threshold_results.append({
            "Model": name,
            "Threshold": threshold,
            "Precision": precision_score(
                y_train,
                predictions,
                zero_division=0,
            ),
            "Recall": recall_score(
                y_train,
                predictions,
                zero_division=0,
            ),
            "F1_Score": f1_score(
                y_train,
                predictions,
                zero_division=0,
            ),
            "False_Positives": fp,
            "False_Negatives": fn,
            "Business_Cost": cost,
        })

# ==========================================
# 5. SELECT MODEL AND THRESHOLD
# ==========================================

cv_df = pd.DataFrame(cv_results)

cv_df = cv_df.sort_values(
    "CV_ROC_AUC",
    ascending=False,
)

threshold_df = pd.DataFrame(threshold_results)

# Illustrative business assumption:
# Missing a churner costs 5 times as much
# as unnecessarily flagging a customer.
best_row = threshold_df.sort_values(
    ["Business_Cost", "F1_Score"],
    ascending=[True, False],
).iloc[0]

best_model_name = best_row["Model"]
best_threshold = float(best_row["Threshold"])

print("\n========== MODEL SELECTION ==========")
print("Selected Model:", best_model_name)
print("Optimal Threshold:", best_threshold)
print(
    "Estimated CV Business Cost:",
    int(best_row["Business_Cost"]),
)

# ==========================================
# 6. TRAIN SELECTED MODEL
# ==========================================

best_model = create_pipeline(
    models[best_model_name]
)

best_model.fit(X_train, y_train)

# ==========================================
# 7. FINAL TEST EVALUATION
# ==========================================

test_probabilities = best_model.predict_proba(
    X_test
)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1],
).ravel()

test_cost = (
    COST_FALSE_NEGATIVE * fn
    + COST_FALSE_POSITIVE * fp
)

print("\n========== FINAL TEST RESULTS ==========")

print(
    "ROC-AUC:",
    round(
        roc_auc_score(y_test, test_probabilities),
        4,
    ),
)
print(
    "Precision:",
    round(
        precision_score(
            y_test,
            test_predictions,
            zero_division=0,
        ),
        4,
    ),
)
print(
    "Recall:",
    round(
        recall_score(
            y_test,
            test_predictions,
            zero_division=0,
        ),
        4,
    ),
)
print(
    "F1 Score:",
    round(
        f1_score(
            y_test,
            test_predictions,
            zero_division=0,
        ),
        4,
    ),
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions))

print("\nFalse Positives:", fp)
print("False Negatives:", fn)
print("Illustrative Business Cost:", test_cost)

# ==========================================
# 8. SAVE RESULTS
# ==========================================

cv_df.to_csv(
    REPORT_DIR / "cross_validation_results.csv",
    index=False,
)

threshold_df.to_csv(
    REPORT_DIR / "threshold_optimization.csv",
    index=False,
)

joblib.dump(
    {
        "model": best_model,
        "threshold": best_threshold,
        "model_name": best_model_name,
        "feature_columns": X.columns.tolist(),
        "cost_assumptions": {
            "false_negative": COST_FALSE_NEGATIVE,
            "false_positive": COST_FALSE_POSITIVE,
        },
    },
    MODEL_DIR / "optimized_churn_model.pkl",
)

print("\n========== OPTIMIZATION COMPLETED ==========")
print("Optimized model saved successfully!")
print("File: models/optimized_churn_model.pkl")