import sys
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import joblib

# ✅ THIS LINE FIXES YOUR ERROR
from src.feature_engineering import create_features

# Load trained model
MODEL_PATH = "models/best_model.pkl"
model = joblib.load(MODEL_PATH)

# Page Config

st.set_page_config(
    page_title="AI Salary Predictor",
    page_icon="💼",
    layout="centered"
)

st.title("💼 AI Salary Prediction System")
st.write("Predict salaries using Machine Learning regression models.")


# User Inputs

st.header("Enter Employee Details")

work_year = st.selectbox(
    "Work Year",
    [2020, 2021, 2022, 2023]
)

employment_type = st.selectbox(
    "Employment Type",
    ["FT", "PT", "CT", "FL"]
)

job_role = st.selectbox(
    "Job Role",
    ["Data Scientist", "ML Engineer", "Data Analyst", "Manager", "Other"]
)

company_size = st.selectbox(
    "Company Size",
    ["S", "M", "L"]
)

location = st.selectbox(
    "Company Location",
    ["HighIncomeCountry", "OtherCountry"]
)

remote_ratio = st.slider(
    "Remote Work Ratio (%)",
    0, 100, 50
)

years_experience = st.slider(
    "Years of Experience",
    0, 15, 3
)

skills_score = st.slider(
    "Skills Score",
    0, 100, 50
)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict Salary"):

    input_data = {
        "WorkYear": work_year,
        "EmploymentType": employment_type,
        "JobRole": job_role,
        "CompanySize": company_size,
        "Location": location,
        "RemoteRatio": remote_ratio,
        "YearsExperience": years_experience,
        "SkillsScore": skills_score
    }

    input_df = pd.DataFrame([input_data])

    # Apply feature engineering
    input_df = create_features(input_df)

    prediction = model.predict(input_df)[0]

    st.success(f"💰 Predicted Salary: ${prediction:,.2f}")

    st.info("Prediction generated using the best trained regression model.")