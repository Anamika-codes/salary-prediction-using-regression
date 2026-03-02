import joblib
import pandas as pd

from feature_engineering import create_features

MODEL_PATH = "models/best_model.pkl"

model = joblib.load(MODEL_PATH)


def predict_salary(employee_data: dict):

    # Convert input to DataFrame
    input_df = pd.DataFrame([employee_data])

    # Apply SAME feature engineering used during training
    input_df = create_features(input_df)

    # Predict
    prediction = model.predict(input_df)

    return float(prediction[0])


if __name__ == "__main__":

    sample_employee = {
        "WorkYear": 2023,
        "EmploymentType": "FT",
        "JobRole": "Data Scientist",
        "CompanySize": "M",
        "Location": "HighIncomeCountry",
        "RemoteRatio": 50,
        "YearsExperience": 5,
        "SkillsScore": 50
    }

    salary = predict_salary(sample_employee)

    print("\nPredicted Salary:", round(salary, 2))