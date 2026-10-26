import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "cleaned_customer_churn.csv"
MODEL_PATH = BASE_DIR / "models" / "optimized_churn_model.pkl"
CV_PATH = BASE_DIR / "reports" / "cross_validation_results.csv"

st.set_page_config(
    page_title="Customer Intelligence Studio",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PURPLE = "#8B5CF6"
LAVENDER = "#A78BFA"
PEACH = "#FFB99A"
GREEN = "#35B89B"
RED = "#EE7991"
DARK = "#29243A"

# ============================================================
# DESIGN
# ============================================================

st.markdown(
    """
<style>
.stApp {
    background:
        radial-gradient(circle at 95% 5%, #F0E8FF, transparent 35%),
        radial-gradient(circle at 5% 45%, #FFF0E6, transparent 35%),
        #FFFCF8;
    color: #29243A;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"],
[data-testid="collapsedControl"] {
    display: none;
}

h1, h2, h3, h4 {
    color: #29243A !important;
}

p, label {
    color: #494158;
}

.brand {
    background: white;
    border: 1px solid #EEE6F5;
    border-radius: 22px;
    padding: 23px 28px;
    margin-bottom: 18px;
}

.brand-name {
    font-size: 25px;
    font-weight: 800;
    color: #29243A;
}

.brand-caption {
    color: #8B79A0;
    font-size: 11px;
    letter-spacing: 2px;
    margin-top: 5px;
}

.hero {
    background: linear-gradient(110deg, #E9DDFF, #F8E2EA, #FFE7D6);
    border-radius: 26px;
    padding: 36px;
    margin: 22px 0 28px;
}

.hero-label {
    color: #7C3AED;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
    margin-bottom: 14px;
}

.hero-title {
    color: #29243A;
    font-size: 37px;
    font-weight: 800;
    line-height: 1.2;
    margin-bottom: 12px;
}

.hero-description {
    color: #5F566F;
    font-size: 16px;
    line-height: 1.7;
    max-width: 760px;
}

.insight-card {
    background: white;
    border: 1px solid #EEE6F5;
    border-radius: 20px;
    padding: 25px;
    min-height: 160px;
    margin: 10px 0;
}

.insight-title {
    color: #29243A;
    font-size: 17px;
    font-weight: 750;
    margin-bottom: 12px;
}

.insight-description {
    color: #696278;
    font-size: 14px;
    line-height: 1.7;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #EEE6F5;
    border-radius: 19px;
    padding: 20px;
}

[data-testid="stMetricLabel"] {
    color: #696278;
}

[data-testid="stMetricValue"] {
    color: #29243A;
    font-weight: 800;
}

[data-testid="stForm"] {
    background: white;
    border: 1px solid #EEE6F5;
    border-radius: 22px;
    padding: 24px;
}

[data-testid="stPlotlyChart"] {
    background: white;
    border: 1px solid #EEE6F5;
    border-radius: 20px;
    padding: 12px;
}

.stButton > button,
.stDownloadButton > button,
button[kind="formSubmit"] {
    background: #8B5CF6 !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #6D28D9 !important;
    color: white !important;
}

div[role="radiogroup"] {
    background: white;
    border: 1px solid #EEE6F5;
    border-radius: 16px;
    padding: 12px;
    gap: 12px;
}

div[role="radiogroup"] label {
    background: #F6F0FF;
    border-radius: 10px;
    padding: 9px 13px;
}

div[role="radiogroup"] label p {
    color: #29243A !important;
    font-weight: 650;
}

div[role="radiogroup"] label[data-checked="true"] {
    background: #E9DDFF;
}

div[role="radiogroup"] label[data-checked="true"] p {
    color: #6D28D9 !important;
}

.footer {
    text-align: center;
    color: #8A8298;
    font-size: 12px;
    line-height: 1.9;
    padding: 24px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# LOAD DATA AND MODEL
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    df = load_data()
    package = load_model()
except Exception as error:
    st.error(f"Unable to load project files: {error}")
    st.stop()

model = package["model"]
threshold = float(package["threshold"])
model_name = package["model_name"]
feature_columns = package["feature_columns"]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def whole_number(value):
    return f"{value:,.0f}"


def whole_percent(value):
    return f"{value:.0%}"


def hero(label, title, description):
    st.markdown(
        '<div class="hero">'
        f'<div class="hero-label">{label}</div>'
        f'<div class="hero-title">{title}</div>'
        f'<div class="hero-description">{description}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def insight(title, description):
    st.markdown(
        '<div class="insight-card">'
        f'<div class="insight-title">{title}</div>'
        f'<div class="insight-description">{description}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def chart_style(fig, height=390):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=30, r=30, t=70, b=40),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Arial",
            size=13,
            color=DARK,
        ),
        title=dict(
            font=dict(size=18, color=DARK),
        ),
        legend=dict(
            font=dict(color=DARK),
            title_font=dict(color=DARK),
        ),
        xaxis=dict(
            tickfont=dict(color=DARK),
            title_font=dict(color=DARK),
            gridcolor="#F0EAF5",
        ),
        yaxis=dict(
            tickfont=dict(color=DARK),
            title_font=dict(color=DARK),
            gridcolor="#F0EAF5",
        ),
    )
    return fig


def risk_category(probability):
    if probability >= 0.70:
        return "High Risk"
    if probability >= threshold:
        return "Medium Risk"
    return "Low Risk"


@st.cache_data
def customer_risk_scores(_model, dataset, columns, cutoff):

    probabilities = _model.predict_proba(
        dataset[columns]
    )[:, 1]

    result = dataset[
        ["customerID", "Contract", "tenure", "MonthlyCharges", "Churn"]
    ].copy()

    result["MonthlyCharges"] = (
        result["MonthlyCharges"].round().astype(int)
    )

    result["Churn Probability"] = probabilities

    result["Risk Category"] = np.select(
        [
            probabilities >= max(0.70, cutoff),
            probabilities >= cutoff,
        ],
        ["High Risk", "Medium Risk"],
        default="Low Risk",
    )

    return result


# ============================================================
# EXPLAINABLE AI
# ============================================================

def professional_feature_name(name):
    """Convert encoded feature names into readable labels."""

    name = str(name)

    for prefix in (
        "categorical__",
        "numerical__",
        "cat__",
        "num__",
    ):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    feature_labels = {
        "gender": "Gender",
        "SeniorCitizen": "Senior Citizen",
        "Partner": "Partner",
        "Dependents": "Dependents",
        "tenure": "Customer Tenure (Months)",
        "PhoneService": "Phone Service",
        "MultipleLines": "Multiple Lines",
        "InternetService": "Internet Service",
        "OnlineSecurity": "Online Security",
        "OnlineBackup": "Online Backup",
        "DeviceProtection": "Device Protection",
        "TechSupport": "Technical Support",
        "StreamingTV": "Streaming TV",
        "StreamingMovies": "Streaming Movies",
        "Contract": "Contract Type",
        "PaperlessBilling": "Paperless Billing",
        "PaymentMethod": "Payment Method",
        "MonthlyCharges": "Monthly Charges",
        "TotalCharges": "Total Charges",
    }

    # Match longer names first to avoid partial matches.
    for original in sorted(feature_labels, key=len, reverse=True):
        if name == original:
            return feature_labels[original]

        if name.startswith(original + "_"):
            category = name[len(original) + 1:]
            return f"{feature_labels[original]}: {category}"

    return name.replace("_", " ")


@st.cache_resource
def get_shap_explainer(_model, _dataset, columns):

    preprocessor = _model.named_steps["preprocessor"]
    classifier = _model.named_steps["classifier"]

    X = _dataset[columns]
    y = _dataset["Churn_Label"].astype(int)

    X_train, _, _, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    background = X_train.sample(
        n=min(100, len(X_train)),
        random_state=42,
    )

    transformed_background = preprocessor.transform(background)

    if hasattr(transformed_background, "toarray"):
        transformed_background = transformed_background.toarray()

    transformed_background = np.asarray(
        transformed_background
    )

    explainer = shap.LinearExplainer(
        classifier,
        transformed_background,
    )

    return explainer


def explain_prediction(customer_data, top_n=10):

    preprocessor = model.named_steps["preprocessor"]

    transformed_customer = preprocessor.transform(
        customer_data
    )

    if hasattr(transformed_customer, "toarray"):
        transformed_customer = transformed_customer.toarray()

    transformed_customer = np.asarray(transformed_customer)

    feature_names = preprocessor.get_feature_names_out()

    explainer = get_shap_explainer(
        model,
        df,
        feature_columns,
    )

    shap_values = explainer(transformed_customer)

    values = np.asarray(shap_values.values[0]).reshape(-1)

    explanation = pd.DataFrame({
        "Feature": [
            professional_feature_name(name)
            for name in feature_names
        ],
        "SHAP Value": values,
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
        .copy()
    )

    return explanation


def display_shap_explanation(explanation):

    st.divider()
    st.subheader("✦ Explainable AI")

    st.write(
        "Understand which customer characteristics "
        "contributed most to the predicted churn risk."
    )

    chart_data = explanation.sort_values(
        "SHAP Value"
    ).copy()

    fig = px.bar(
        chart_data,
        x="SHAP Value",
        y="Feature",
        color="Impact",
        orientation="h",
        color_discrete_map={
            "Increases Churn Risk": RED,
            "Decreases Churn Risk": GREEN,
        },
        title="Customer Churn Risk Drivers",
    )

    fig.update_traces(
        hovertemplate=(
            "Customer Characteristic: %{y}<br>"
            "Contribution (Log-Odds): %{x:.3f}"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        template="plotly_white",
        height=550,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Arial",
            size=14,
            color=DARK,
        ),
        title=dict(
            text="Customer Churn Risk Drivers",
            font=dict(
                size=20,
                color=DARK,
            ),
        ),
        xaxis=dict(
            title="Contribution to Churn Prediction (Log-Odds)",
            title_font=dict(color=DARK),
            tickfont=dict(color=DARK),
            gridcolor="#F0EAF5",
            zeroline=True,
            zerolinecolor="#777083",
        ),
        yaxis=dict(
            title=None,
            tickfont=dict(
                color=DARK,
                size=12,
            ),
            automargin=True,
        ),
        legend=dict(
            title="Prediction Impact",
            font=dict(color=DARK),
            title_font=dict(color=DARK),
            orientation="h",
            y=-0.25,
            x=0,
        ),
        margin=dict(
            l=45,
            r=35,
            t=75,
            b=95,
        ),
    )

    st.plotly_chart(
        fig,
        width="stretch",
        theme=None,
    )

    st.caption(
        "Positive SHAP values increase the model's churn score; "
        "negative values decrease it. Contributions are measured "
        "in log-odds, not percentages. These explanations describe "
        "model behaviour and do not establish causation."
    )


# ============================================================
# HEADER AND NAVIGATION
# ============================================================

st.markdown(
    '<div class="brand">'
    '<div class="brand-name">✦ Customer Intelligence Studio</div>'
    '<div class="brand-caption">'
    'AI-POWERED CUSTOMER CHURN PREDICTION'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

page = st.radio(
    "Navigation",
    [
        "Overview",
        "Churn Prediction",
        "Customer Analytics",
        "Risk Intelligence",
        "Model Lab",
    ],
    horizontal=True,
    label_visibility="collapsed",
)

# ============================================================
# PAGE 1: OVERVIEW
# ============================================================

if page == "Overview":

    hero(
        "CUSTOMER ANALYTICS / OVERVIEW",
        "Understand customers.<br>Predict what comes next.",
        "Transform customer data into meaningful retention insights "
        "using predictive analytics and machine learning.",
    )

    total = len(df)
    churned = int(df["Churn_Label"].sum())
    retained = total - churned
    churn_rate = df["Churn_Label"].mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Customers", whole_number(total))
    c2.metric("Retained Customers", whole_number(retained))
    c3.metric("Churned Customers", whole_number(churned))
    c4.metric("Customer Churn Rate", whole_percent(churn_rate))

    st.write("")

    left, right = st.columns([1, 1.3])

    with left:

        counts = df["Churn"].value_counts().reset_index()
        counts.columns = ["Status", "Customers"]

        fig = px.pie(
            counts,
            names="Status",
            values="Customers",
            hole=0.7,
            title="Customer Retention Mix",
            color="Status",
            color_discrete_map={
                "No": LAVENDER,
                "Yes": PEACH,
            },
        )

        fig.update_traces(
            texttemplate="%{percent:.0%}",
            hovertemplate=(
                "%{label}<br>"
                "Customers: %{value:,.0f}<br>"
                "Share: %{percent:.0%}<extra></extra>"
            ),
        )

        st.plotly_chart(
            chart_style(fig),
            width="stretch",
            theme=None,
        )

    with right:

        contracts = (
            df.groupby("Contract")["Churn_Label"]
            .mean()
            .mul(100)
            .reset_index()
        )

        contracts.columns = ["Contract", "Churn Rate"]

        fig = px.bar(
            contracts,
            x="Contract",
            y="Churn Rate",
            color="Contract",
            title="Churn Rate by Contract Type",
            color_discrete_sequence=[
                LAVENDER,
                PEACH,
                GREEN,
            ],
        )

        fig.update_traces(
            texttemplate="%{y:.0f}%",
            textposition="outside",
            hovertemplate=(
                "Contract: %{x}<br>"
                "Churn Rate: %{y:.0f}%<extra></extra>"
            ),
        )

        fig.update_layout(showlegend=False)

        st.plotly_chart(
            chart_style(fig),
            width="stretch",
            theme=None,
        )

    st.subheader("Business Intelligence Highlights")

    a, b, c = st.columns(3)

    with a:
        insight(
            "✦ Contract Intelligence",
            "Month-to-month customers show substantially higher "
            "historical churn rates than customers with longer contracts.",
        )

    with b:
        insight(
            "✦ Payment Behaviour",
            "Electronic check users demonstrate higher historical "
            "churn rates compared with automatic payment users.",
        )

    with c:
        insight(
            "✦ Customer Lifecycle",
            "Customers who churn generally have shorter tenure "
            "and higher average monthly charges.",
        )

# ============================================================
# PAGE 2: CHURN PREDICTION
# ============================================================

elif page == "Churn Prediction":

    hero(
        "PREDICTIVE INTELLIGENCE",
        "Customer Churn Assessment",
        "Enter customer information to estimate churn probability "
        "using the optimized machine learning model.",
    )

    with st.form("prediction_form"):

        st.subheader("Personal & Account Details")

        c1, c2, c3 = st.columns(3)

        with c1:
            gender = st.selectbox(
                "Gender",
                ["Female", "Male"],
            )

            senior = st.selectbox(
                "Senior Citizen",
                ["No", "Yes"],
            )

            partner = st.selectbox(
                "Partner",
                ["No", "Yes"],
            )

        with c2:
            dependents = st.selectbox(
                "Dependents",
                ["No", "Yes"],
            )

            tenure = st.slider(
                "Tenure (Months)",
                min_value=0,
                max_value=72,
                value=12,
                step=1,
            )

            phone = st.selectbox(
                "Phone Service",
                ["Yes", "No"],
            )

        with c3:
            internet = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"],
            )

            contract = st.selectbox(
                "Contract Type",
                ["Month-to-month", "One year", "Two year"],
            )

            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        st.divider()
        st.subheader("Services & Subscription")

        s1, s2, s3 = st.columns(3)

        with s1:
            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"],
            )

            security = st.selectbox(
                "Online Security",
                ["No", "Yes", "No internet service"],
            )

            backup = st.selectbox(
                "Online Backup",
                ["No", "Yes", "No internet service"],
            )

        with s2:
            protection = st.selectbox(
                "Device Protection",
                ["No", "Yes", "No internet service"],
            )

            support = st.selectbox(
                "Tech Support",
                ["No", "Yes", "No internet service"],
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                ["No", "Yes", "No internet service"],
            )

        with s3:
            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["No", "Yes", "No internet service"],
            )

            paperless = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"],
            )

        st.divider()
        st.subheader("Billing Information")

        b1, b2 = st.columns(2)

        with b1:
            monthly = st.number_input(
                "Monthly Charges",
                min_value=0,
                max_value=500,
                value=71,
                step=1,
                format="%d",
            )

        with b2:
            total_charges = st.number_input(
                "Total Charges",
                min_value=0,
                max_value=20000,
                value=840,
                step=1,
                format="%d",
            )

        submitted = st.form_submit_button(
            "✦ Generate Churn Prediction",
            width="stretch",
        )

    if submitted:

        if phone == "No":
            multiple_lines = "No phone service"

        elif multiple_lines == "No phone service":
            multiple_lines = "No"

        services = [
            security,
            backup,
            protection,
            support,
            streaming_tv,
            streaming_movies,
        ]

        if internet == "No":
            services = ["No internet service"] * 6

        else:
            services = [
                "No" if item == "No internet service" else item
                for item in services
            ]

        (
            security,
            backup,
            protection,
            support,
            streaming_tv,
            streaming_movies,
        ) = services

        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total_charges,
        }])

        customer = customer[feature_columns]

        probability = float(
            model.predict_proba(customer)[0][1]
        )

        risk = risk_category(probability)

        outcome = (
            "Likely to Churn"
            if probability >= threshold
            else "Likely to Stay"
        )

        st.divider()
        st.subheader("Prediction Results")

        r1, r2, r3 = st.columns(3)

        r1.metric(
            "Churn Probability",
            whole_percent(probability),
        )

        r2.metric(
            "Predicted Outcome",
            outcome,
        )

        r3.metric(
            "Risk Category",
            risk,
        )

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={
                    "suffix": "%",
                    "valueformat": ".0f",
                },
                title={
                    "text": "Estimated Churn Probability",
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickformat": ".0f",
                    },
                    "bar": {
                        "color": PURPLE,
                    },
                    "steps": [
                        {
                            "range": [
                                0,
                                threshold * 100,
                            ],
                            "color": "#DDF8EC",
                        },
                        {
                            "range": [
                                threshold * 100,
                                max(70, threshold * 100),
                            ],
                            "color": "#FFF0D8",
                        },
                        {
                            "range": [
                                max(70, threshold * 100),
                                100,
                            ],
                            "color": "#FFE5E9",
                        },
                    ],
                },
            )
        )

        st.plotly_chart(
            chart_style(fig, 350),
            width="stretch",
            theme=None,
        )

        if risk == "High Risk":
            st.error(
                "High risk: Consider prioritizing this customer "
                "for retention outreach."
            )

        elif risk == "Medium Risk":
            st.warning(
                "Medium risk: Review the customer's subscription "
                "experience and satisfaction."
            )

        else:
            st.success(
                "Low risk: Continue monitoring customer engagement."
            )

        st.caption(
            "Predictions are estimates, not guaranteed outcomes. "
            "Risk categories are illustrative."
        )

        # SHAP EXPLANATION

        try:
            explanation = explain_prediction(customer)
            display_shap_explanation(explanation)

        except Exception as error:
            st.warning(
                "The prediction was successful, but the "
                f"AI explanation could not be generated: {error}"
            )

# ============================================================
# PAGE 3: CUSTOMER ANALYTICS
# ============================================================

elif page == "Customer Analytics":

    hero(
        "EXPLORATORY ANALYTICS",
        "Customer Behaviour Explorer",
        "Explore subscription patterns, customer tenure, "
        "billing characteristics, and historical churn.",
    )

    options = {
        "Contract Type": "Contract",
        "Payment Method": "PaymentMethod",
        "Internet Service": "InternetService",
        "Tech Support": "TechSupport",
        "Online Security": "OnlineSecurity",
        "Paperless Billing": "PaperlessBilling",
    }

    selected = st.selectbox(
        "Explore churn by",
        list(options.keys()),
    )

    column = options[selected]

    summary = (
        df.groupby(column)["Churn_Label"]
        .agg(["mean", "count"])
        .reset_index()
    )

    summary["Churn Rate"] = (
        summary["mean"] * 100
    ).round().astype(int)

    summary = summary.rename(
        columns={
            "count": "Customers",
            column: selected,
        }
    )

    fig = px.bar(
        summary,
        x=selected,
        y="Churn Rate",
        color="Churn Rate",
        color_continuous_scale=[
            "#E9DDFF",
            "#B794F6",
            "#7C3AED",
        ],
        title=f"Churn Rate by {selected}",
    )

    fig.update_traces(
        texttemplate="%{y:.0f}%",
        textposition="outside",
        hovertemplate=(
            "%{x}<br>"
            "Churn Rate: %{y:.0f}%<extra></extra>"
        ),
    )

    st.plotly_chart(
        chart_style(fig, 430),
        width="stretch",
        theme=None,
    )

    st.subheader("Customer Segment Statistics")

    st.dataframe(
        summary[[selected, "Customers", "Churn Rate"]],
        width="stretch",
        hide_index=True,
        column_config={
            "Churn Rate": st.column_config.NumberColumn(
                "Churn Rate (%)",
                format="%d%%",
            ),
        },
    )

    left, right = st.columns(2)

    with left:

        fig = px.box(
            df,
            x="Churn",
            y="MonthlyCharges",
            color="Churn",
            title="Monthly Charges by Churn Status",
            labels={
                "Churn": "Customer Churn Status",
                "MonthlyCharges": "Monthly Charges",
            },
            color_discrete_map={
                "No": LAVENDER,
                "Yes": PEACH,
            },
        )

        fig.update_traces(
            hovertemplate=(
                "Monthly Charges: %{y:,.0f}"
                "<extra></extra>"
            )
        )

        st.plotly_chart(
            chart_style(fig),
            width="stretch",
            theme=None,
        )

    with right:

        fig = px.histogram(
            df,
            x="tenure",
            color="Churn",
            nbins=30,
            barmode="overlay",
            title="Customer Tenure Distribution",
            labels={
                "tenure": "Customer Tenure (Months)",
                "count": "Number of Customers",
                "Churn": "Customer Churn Status",
            },
            color_discrete_map={
                "No": LAVENDER,
                "Yes": PEACH,
            },
        )

        fig.update_yaxes(
            title_text="Number of Customers"
        )

        fig.update_traces(
            hovertemplate=(
                "Tenure: %{x:.0f} months<br>"
                "Customers: %{y:.0f}<extra></extra>"
            )
        )

        st.plotly_chart(
            chart_style(fig),
            width="stretch",
            theme=None,
        )

# ============================================================
# PAGE 4: RISK INTELLIGENCE
# ============================================================

elif page == "Risk Intelligence":

    hero(
        "CUSTOMER RISK INTELLIGENCE",
        "Customer Risk Segmentation",
        "Explore model-estimated churn probabilities "
        "and identify customer groups for further review.",
    )

    risk_df = customer_risk_scores(
        model,
        df,
        feature_columns,
        threshold,
    )

    counts = risk_df[
        "Risk Category"
    ].value_counts()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "High Risk",
        whole_number(counts.get("High Risk", 0)),
    )

    c2.metric(
        "Medium Risk",
        whole_number(counts.get("Medium Risk", 0)),
    )

    c3.metric(
        "Low Risk",
        whole_number(counts.get("Low Risk", 0)),
    )

    left, right = st.columns(2)

    with left:

        risk_counts = counts.reset_index()
        risk_counts.columns = [
            "Risk Category",
            "Customers",
        ]

        fig = px.pie(
            risk_counts,
            names="Risk Category",
            values="Customers",
            hole=0.65,
            title="Risk Segment Distribution",
            color="Risk Category",
            color_discrete_map={
                "High Risk": RED,
                "Medium Risk": PEACH,
                "Low Risk": GREEN,
            },
        )

        fig.update_traces(
            texttemplate="%{percent:.0%}",
            hovertemplate=(
                "%{label}<br>"
                "Customers: %{value:,.0f}<br>"
                "Share: %{percent:.0%}<extra></extra>"
            ),
        )

        st.plotly_chart(
            chart_style(fig),
            width="stretch",
            theme=None,
        )

    with right:

        fig = px.histogram(
            risk_df,
            x="Churn Probability",
            nbins=35,
            title="Estimated Churn Probability Distribution",
            color_discrete_sequence=[LAVENDER],
        )

        fig.update_yaxes(
            title_text="Number of Customers"
        )

        fig.update_traces(
            hovertemplate=(
                "Churn Probability: %{x:.0%}<br>"
                "Customers: %{y:.0f}<extra></extra>"
            ),
        )

        styled_fig = chart_style(fig)
        styled_fig.update_xaxes(tickformat=".0%")

        st.plotly_chart(
            styled_fig,
            width="stretch",
            theme=None,
        )

    st.subheader("Customer Risk Explorer")

    selected_risk = st.selectbox(
        "Filter customers",
        [
            "All Customers",
            "High Risk",
            "Medium Risk",
            "Low Risk",
        ],
    )

    if selected_risk == "All Customers":
        filtered = risk_df.copy()

    else:
        filtered = risk_df[
            risk_df["Risk Category"] == selected_risk
        ].copy()

    filtered = filtered.sort_values(
        "Churn Probability",
        ascending=False,
    )

    display_df = filtered.copy()

    display_df["Churn Probability"] = (
        display_df["Churn Probability"]
        .mul(100)
        .round()
        .astype(int)
        .astype(str) + "%"
    )

    display_df = display_df.rename(
        columns={
            "customerID": "Customer ID",
            "tenure": "Customer Tenure (Months)",
            "MonthlyCharges": "Monthly Charges",
            "Churn": "Actual Churn",
        }
    )

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        "Download Risk Report",
        data=display_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="customer_risk_report.csv",
        mime="text/csv",
    )

    st.caption(
        "These are historical-data model scores, including "
        "training records, not independent validation results."
    )

# ============================================================
# PAGE 5: MODEL LAB
# ============================================================

elif page == "Model Lab":

    hero(
        "MACHINE LEARNING LABORATORY",
        "Model Performance Studio",
        "Review cross-validation results, model selection, "
        "and final held-out test performance.",
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Selected Algorithm",
        model_name,
    )

    c2.metric(
        "Decision Threshold",
        whole_percent(threshold),
    )

    if CV_PATH.exists():

        cv_df = pd.read_csv(CV_PATH)

        st.subheader("Cross-Validation Results")

        display_cv = cv_df.copy()

        for column in [
            "CV_ROC_AUC",
            "CV_ROC_AUC_STD",
            "CV_Average_Precision",
        ]:
            if column in display_cv.columns:
                display_cv[column] = (
                    display_cv[column]
                    .mul(100)
                    .round()
                    .astype(int)
                    .astype(str) + "%"
                )

        display_cv = display_cv.rename(
            columns={
                "Model": "Machine Learning Model",
                "CV_ROC_AUC": "ROC-AUC",
                "CV_ROC_AUC_STD": "ROC-AUC Variation",
                "CV_Average_Precision": "Average Precision",
            }
        )

        st.dataframe(
            display_cv,
            width="stretch",
            hide_index=True,
        )

        chart_cv = cv_df.copy()

        chart_cv["ROC-AUC Percentage"] = (
            chart_cv["CV_ROC_AUC"] * 100
        )

        fig = px.bar(
            chart_cv,
            x="Model",
            y="ROC-AUC Percentage",
            color="Model",
            title="Cross-Validation ROC-AUC Comparison",
            color_discrete_sequence=[
                LAVENDER,
                PEACH,
                GREEN,
            ],
        )

        fig.update_traces(
            texttemplate="%{y:.0f}%",
            textposition="outside",
            hovertemplate=(
                "%{x}<br>"
                "ROC-AUC: %{y:.0f}%<extra></extra>"
            ),
        )

        fig.update_layout(
            showlegend=False,
            yaxis_range=[0, 100],
        )

        st.plotly_chart(
            chart_style(fig),
            width="stretch",
            theme=None,
        )

    st.divider()
    st.subheader("Final Test Evaluation")

    X = df[feature_columns]
    y = df["Churn_Label"].astype(int)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    auc = roc_auc_score(
        y_test,
        probabilities,
    )

    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric(
        "Accuracy",
        whole_percent(accuracy),
    )

    m2.metric(
        "Precision",
        whole_percent(precision),
    )

    m3.metric(
        "Recall",
        whole_percent(recall),
    )

    m4.metric(
        "F1 Score",
        whole_percent(f1),
    )

    m5.metric(
        "ROC-AUC",
        whole_percent(auc),
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    fig = px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale=[
            "#F4ECFF",
            "#A78BFA",
            "#7C3AED",
        ],
        x=[
            "Predicted Stay",
            "Predicted Churn",
        ],
        y=[
            "Actual Stay",
            "Actual Churn",
        ],
        title="Confusion Matrix",
    )

    fig.update_traces(
        texttemplate="%{z:.0f}",
        hovertemplate=(
            "Actual: %{y}<br>"
            "Predicted: %{x}<br>"
            "Customers: %{z:.0f}<extra></extra>"
        ),
    )

    st.plotly_chart(
        chart_style(fig),
        width="stretch",
        theme=None,
    )

    st.info(
        "The model prioritizes detecting customers likely to churn. "
        "This improves recall but increases false positives."
    )

    st.caption(
        "The threshold was selected using training-data "
        "cross-validation and illustrative business-cost assumptions."
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    '<div class="footer">'
    '✦ CUSTOMER INTELLIGENCE STUDIO<br>'
    'AI-Powered Customer Churn Prediction<br>'
    'Python · Scikit-learn · Streamlit · Plotly · SHAP'
    '</div>',
    unsafe_allow_html=True,
)