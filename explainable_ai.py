import numpy as np
import pandas as pd
import shap
import streamlit as st
import plotly.express as px


def explain_prediction(model, customer_data, top_n=10):
    """
    Explain an individual churn prediction using SHAP.
    Compatible with a scikit-learn Pipeline containing
    a preprocessor and Logistic Regression classifier.
    """

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    transformed_data = preprocessor.transform(customer_data)

    if hasattr(transformed_data, "toarray"):
        transformed_data = transformed_data.toarray()

    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.LinearExplainer(classifier, transformed_data)

    shap_values = explainer.shap_values(transformed_data)

    explanation = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values[0],
    })

    explanation["Impact"] = np.where(
        explanation["SHAP Value"] > 0,
        "Increases Churn Risk",
        "Decreases Churn Risk",
    )

    explanation["Absolute Impact"] = (
        explanation["SHAP Value"].abs()
    )

    explanation = (
        explanation.sort_values(
            "Absolute Impact",
            ascending=False,
        )
        .head(top_n)
    )

    explanation["Feature"] = (
        explanation["Feature"]
        .str.replace("cat__", "", regex=False)
        .str.replace("num__", "", regex=False)
        .str.replace("_", " ")
    )

    return explanation


def display_shap_explanation(explanation):

    st.subheader("AI Prediction Explanation")

    st.write(
        "Understand which customer characteristics "
        "contributed most to this prediction."
    )

    fig = px.bar(
        explanation.sort_values("SHAP Value"),
        x="SHAP Value",
        y="Feature",
        color="Impact",
        orientation="h",
        color_discrete_map={
            "Increases Churn Risk": "#EE7991",
            "Decreases Churn Risk": "#35B89B",
        },
        title="Customer Churn Risk Drivers",
    )

    fig.update_layout(
        template="plotly_white",
        height=450,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#29243A"),
        legend_title_text="Prediction Impact",
        xaxis_title="SHAP Contribution",
        yaxis_title="Customer Characteristic",
    )

    st.plotly_chart(fig, width="stretch")

    st.caption(
        "Positive SHAP values increase the model's churn score. "
        "Negative values decrease it. "
        "These are model explanations, not causal effects."
    )