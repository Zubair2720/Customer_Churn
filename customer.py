
import streamlit as st
import pandas as pd
import joblib

# Load Model and Scaler
model = joblib.load('gb.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer will leave the service or not.")

st.divider()

# Customer Input Form
st.subheader("Enter Customer Details")

with st.form("customer_form"):

    age = st.number_input("Age", min_value=18, max_value=100, value=25)

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    tenure = st.number_input("Tenure", min_value=0, value=12)

    usage = st.number_input(
        "Usage Frequency", min_value=0, value=15
    )

    support = st.number_input(
        "Support Calls", min_value=0, value=2
    )

    payment = st.number_input(
        "Payment Delay", min_value=0, value=5
    )

    subscription = st.selectbox(
        "Subscription Type",
        ["Basic", "Standard", "Premium"]
    )

    contract = st.selectbox(
        "Contract Length",
        ["Monthly", "Quarterly", "Annual"]
    )

    spend = st.number_input(
        "Total Spend", min_value=0.0, value=500.0
    )

    interaction = st.number_input(
        "Last Interaction", min_value=0, value=10
    )

    submit = st.form_submit_button("Predict Churn")

# Prediction
if submit:

    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Tenure": tenure,
        "Usage Frequency": usage,
        "Support Calls": support,
        "Payment Delay": payment,
        "Subscription Type": subscription,
        "Contract Length": contract,
        "Total Spend": spend,
        "Last Interaction": interaction
    }])

    # Gender Encoding
    input_data["Gender"] = input_data["Gender"].map({
        "Male": 0,
        "Female": 1
    })

    # Categorical Encoding
    input_data = pd.get_dummies(
        input_data,
        columns=["Subscription Type", "Contract Length"],
        dtype=int
    )

    # Match Training Columns
    input_data = input_data.reindex(
        columns=columns,
        fill_value=0
    )

    # Feature Scaling
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    st.divider()

    if prediction[0] == 1:
        st.error("⚠️ Customer is likely to Churn!")
    else:
        st.success("✅ Customer is likely to Stay!")

    st.write("Prediction:", int(prediction[0]))
