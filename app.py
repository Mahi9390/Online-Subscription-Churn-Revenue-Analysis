import streamlit as st
import joblib
import numpy as np

# -------------------------------------------------
# Load trained model and scaler
# -------------------------------------------------
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Churn Prediction App", layout="centered")

st.title("📉 Subscription Churn Prediction")
st.write("Enter customer details to estimate churn risk.")

# -------------------------------------------------
# User Inputs
# -------------------------------------------------
accountage = st.number_input("Account Age (months)", min_value=0, max_value=120)
monthlycharges = st.number_input("Monthly Charges", min_value=0.0)
totalcharges = st.number_input("Total Charges", min_value=0.0)

viewinghoursperweek = st.number_input("Viewing Hours per Week", min_value=0.0)
averageviewingduration = st.number_input("Average Viewing Duration (minutes)", min_value=0.0)
contentdownloadspermonth = st.number_input("Content Downloads per Month", min_value=0)

supportticketspermonth = st.number_input("Support Tickets per Month", min_value=0)
watchlistsize = st.number_input("Watchlist Size", min_value=0)

userrating = st.slider("User Rating (1 = Low, 5 = High)", 1, 5)

auto_renew_flag = st.selectbox(
    "Auto Renew Enabled?",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("Predict Churn"):
    # -------- Numeric features used during training --------
    numeric_input = np.array([[
        accountage,
        monthlycharges,
        totalcharges,
        viewinghoursperweek,
        averageviewingduration,
        contentdownloadspermonth,
        supportticketspermonth,
        watchlistsize,
        viewinghoursperweek / 7,   # avg_daily_usage_hours
        userrating
    ]])

    # Scale numeric features
    numeric_scaled = scaler.transform(numeric_input)

    # Append binary feature (NOT scaled)
    final_input = np.hstack([numeric_scaled, [[auto_renew_flag]]])

    # Predict churn probability
    churn_prob = model.predict_proba(final_input)[0][1]

    # -------------------------------------------------
    # Output
    # -------------------------------------------------
    st.subheader("Prediction Result")
    st.write(f"**Churn Probability:** {churn_prob:.2f}")

    if churn_prob >= 0.5:
        st.error("⚠️ High Churn Risk")
    else:
        st.success("✅ Low Churn Risk")
