import streamlit as st
import pandas as pd
import joblib
from sklearn.compose import _column_transformer


# Older scikit-learn artifacts reference this private list type during loading.
if not hasattr(_column_transformer, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass

    _column_transformer._RemainderColsList = _RemainderColsList


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------------------------------
# LOAD TRAINED MODEL AND OTHER ARTIFACTS
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    return joblib.load("customer_churn_artifacts.pkl")


artifacts = load_artifacts()

logistic_model = artifacts["logistic_model"]
preprocessor = artifacts["preprocessor"]
kmeans_model = artifacts["kmeans_model"]
scaler = artifacts["scaler"]

numerical_features = artifacts["numerical_features"]
categorical_features = artifacts["categorical_features"]
segmentation_features = artifacts["segmentation_features"]


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------
st.title("📊 Customer Churn Prediction System")
st.write(
    "Enter customer information below to predict churn probability "
    "and identify the customer's segment."
)

st.divider()


# =========================================================
# STEP 1: CUSTOMER PROFILE
# =========================================================

st.header("👤 1. Customer Profile")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["Yes", "No"]
    )

with col2:
    married = st.selectbox(
        "Married",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    number_of_dependents = st.number_input(
        "Number of Dependents",
        min_value=0,
        max_value=10,
        value=0
    )

with col3:
    referred_a_friend = st.selectbox(
        "Referred a Friend",
        ["Yes", "No"]
    )

    number_of_referrals = st.number_input(
        "Number of Referrals",
        min_value=0,
        max_value=20,
        value=0
    )

    tenure = st.number_input(
        "Tenure in Months",
        min_value=0,
        max_value=100,
        value=12
    )


# =========================================================
# STEP 2: SERVICE INFORMATION
# =========================================================

st.header("📡 2. Service Information")

col1, col2, col3 = st.columns(3)

with col1:
    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No Phone Service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["Yes", "No"]
    )

with col2:
    internet_type = st.selectbox(
        "Internet Type",
        ["DSL", "Fiber Optic", "Cable", "No Internet"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No Internet Service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No Internet Service"]
    )

with col3:
    device_protection = st.selectbox(
        "Device Protection Plan",
        ["Yes", "No", "No Internet Service"]
    )

    tech_support = st.selectbox(
        "Premium Tech Support",
        ["Yes", "No", "No Internet Service"]
    )

    unlimited_data = st.selectbox(
        "Unlimited Data",
        ["Yes", "No", "No Internet Service"]
    )


col1, col2, col3 = st.columns(3)

with col1:
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No Internet Service"]
    )

with col2:
    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No Internet Service"]
    )

with col3:
    streaming_music = st.selectbox(
        "Streaming Music",
        ["Yes", "No", "No Internet Service"]
    )


# =========================================================
# STEP 3: PLAN & PAYMENT
# =========================================================

st.header("💳 3. Plan & Payment")

col1, col2, col3 = st.columns(3)

with col1:
    contract = st.selectbox(
        "Contract",
        ["Month-to-Month", "One Year", "Two Year"]
    )

with col2:
    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

with col3:
    payment_method = st.selectbox(
        "Payment Method",
        ["Bank Withdrawal", "Credit Card", "Mailed Check"]
    )


offer = st.selectbox(
    "Offer",
    ["No Offer", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"]
)


# =========================================================
# STEP 4: FINANCIAL INFORMATION
# =========================================================

st.header("💰 4. Financial Information")

col1, col2, col3 = st.columns(3)

with col1:
    monthly_charge = st.number_input(
        "Monthly Charge",
        min_value=0.0,
        max_value=2000.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=100000.0,
        value=840.0
    )

    total_revenue = st.number_input(
        "Total Revenue",
        min_value=0.0,
        max_value=100000.0,
        value=1000.0
    )

with col2:
    avg_long_distance = st.number_input(
        "Avg Monthly Long Distance Charges",
        min_value=0.0,
        max_value=200.0,
        value=20.0
    )

    avg_gb_download = st.number_input(
        "Avg Monthly GB Download",
        min_value=0.0,
        max_value=500.0,
        value=20.0
    )

    total_long_distance = st.number_input(
        "Total Long Distance Charges",
        min_value=0.0,
        max_value=100000.0,
        value=200.0
    )

with col3:
    total_refunds = st.number_input(
        "Total Refunds",
        min_value=0.0,
        max_value=10000.0,
        value=0.0
    )

    total_extra_data = st.number_input(
        "Total Extra Data Charges",
        min_value=0.0,
        max_value=10000.0,
        value=0.0
    )

    population = st.number_input(
        "Population",
        min_value=0,
        max_value=1000000,
        value=10000
    )


# =========================================================
# STEP 5: CUSTOMER VALUE
# =========================================================

st.header("⭐ 5. Customer Value")

col1, col2 = st.columns(2)

with col1:
    satisfaction_score = st.slider(
        "Satisfaction Score",
        min_value=1,
        max_value=5,
        value=3
    )

with col2:
    cltv = st.number_input(
        "Customer Lifetime Value (CLTV)",
        min_value=0.0,
        max_value=100000.0,
        value=4000.0
    )


# =========================================================
# PREDICTION
# =========================================================

st.divider()

if st.button("🔮 Predict Customer", type="primary"):

    # Create a DataFrame containing ALL 37 features
    customer_data = pd.DataFrame([{

        # Customer Profile
        "Gender": gender,
        "Age": age,
        "Under 30": "Yes" if age < 30 else "No",
        "Senior Citizen": senior_citizen,
        "Married": married,
        "Dependents": dependents,
        "Number of Dependents": number_of_dependents,
        "Referred a Friend": referred_a_friend,
        "Number of Referrals": number_of_referrals,

        # Service information
        "Tenure in Months": tenure,
        "Offer": offer,
        "Phone Service": phone_service,
        "Avg Monthly Long Distance Charges": avg_long_distance,
        "Multiple Lines": multiple_lines,
        "Internet Service": internet_service,
        "Internet Type": internet_type,
        "Avg Monthly GB Download": avg_gb_download,
        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection Plan": device_protection,
        "Premium Tech Support": tech_support,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies,
        "Streaming Music": streaming_music,
        "Unlimited Data": unlimited_data,

        # Plan
        "Contract": contract,
        "Paperless Billing": paperless_billing,
        "Payment Method": payment_method,

        # Financial
        "Monthly Charge": monthly_charge,
        "Total Charges": total_charges,
        "Total Refunds": total_refunds,
        "Total Extra Data Charges": total_extra_data,
        "Total Long Distance Charges": total_long_distance,
        "Total Revenue": total_revenue,

        # Customer value
        "Satisfaction Score": satisfaction_score,
        "CLTV": cltv,

        # Location-related numerical feature
        "Population": population
    }])


    # -----------------------------------------------------
    # CHURN PREDICTION
    # -----------------------------------------------------

    customer_processed = preprocessor.transform(customer_data)

    churn_prediction = logistic_model.predict(customer_processed)[0]

    churn_probability = logistic_model.predict_proba(
        customer_processed
    )[0][1]


    # -----------------------------------------------------
    # CUSTOMER SEGMENTATION
    # -----------------------------------------------------

    customer_segmentation = customer_data[segmentation_features]

    customer_segmentation_scaled = scaler.transform(
        customer_segmentation
    )

    cluster = kmeans_model.predict(
        customer_segmentation_scaled
    )[0]


    if cluster == 0:
        segment = "Newer / Lower-Value Customer"
    else:
        segment = "High-Value / Established Customer"


    # -----------------------------------------------------
    # DISPLAY RESULTS
    # -----------------------------------------------------

    st.subheader("📊 Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        if churn_prediction == 1:
            st.error("⚠️ Likely to Churn")
        else:
            st.success("✅ Not Likely to Churn")

    with col2:
        st.metric(
            "Churn Probability",
            f"{churn_probability * 100:.2f}%"
        )

    with col3:
        st.metric(
            "Customer Cluster",
            cluster
        )

    st.info(f"**Customer Segment:** {segment}")