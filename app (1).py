import os
import pickle
import pandas as pd
import streamlit as st

# -----------------------------
# Page Configuration (Must be first Streamlit command)
# -----------------------------
st.set_page_config(
    page_title="Car Purchase Prediction",
    page_icon="🚗",
    layout="centered"
)

# -----------------------------
# Load Trained Model Safely
# -----------------------------
@st.cache_resource
def load_model():
    # Resolve dynamic absolute path to the directory containing app.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "car_purchase_model.pkl")
    
    if not os.path.exists(model_path):
        st.error(
            f"🚨 Could not find `car_purchase_model.pkl`.\n\n"
            f"Expected path: `{model_path}`\n\n"
            f"Please make sure `car_purchase_model.pkl` is committed to the same repository directory as `app.py`."
        )
        st.stop()
        
    with open(model_path, "rb") as file:
        return pickle.load(file)

model = load_model()

# -----------------------------
# App Layout & Title
# -----------------------------
st.title("🚗 Car Purchase Prediction")
st.write("Enter customer details below to predict whether the customer will purchase a car.")

# -----------------------------
# User Inputs
# -----------------------------
age = st.number_input("Age", min_value=18, max_value=100, value=30)

annual_salary = st.number_input(
    "Annual Salary",
    min_value=0,
    value=50000,
    step=1000
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

married = st.selectbox(
    "Married",
    ["No", "Yes"]
)

children = st.number_input(
    "Children",
    min_value=0,
    max_value=10,
    value=0
)

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=700
)

car_price = st.number_input(
    "Car Price",
    min_value=1000,
    value=20000,
    step=1000
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0,
    value=5000,
    step=500
)

# -----------------------------
# Encode Categorical Variables
# -----------------------------
gender_encoded = 1 if gender == "Male" else 0
married_encoded = 1 if married == "Yes" else 0

# -----------------------------
# Prepare Input Data
# -----------------------------
input_data = pd.DataFrame({
    "Age": [age],
    "Annual_Salary": [annual_salary],
    "Gender": [gender_encoded],
    "Married": [married_encoded],
    "Children": [children],
    "Credit_Score": [credit_score],
    "Car_Price": [car_price],
    "Loan_Amount": [loan_amount]
})

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Customer is likely to purchase a car.")
    else:
        st.error("❌ Customer is not likely to purchase a car.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.write("Developed using Streamlit & Machine Learning")
