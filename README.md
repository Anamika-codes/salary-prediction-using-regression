# 💼 AI Salary Prediction System Using Regression

An end-to-end Machine Learning project that predicts employee salaries using regression models and provides explainable insights through SHAP along with an interactive Streamlit web application.

This project demonstrates a complete AI pipeline — from data ingestion to deployment — following industry-level ML engineering practices.

---

## 🚀 Project Overview

The goal of this project is to build an intelligent system capable of predicting salaries based on employee and company attributes such as:

- Experience level
- Job role
- Employment type
- Company size
- Location
- Remote work ratio
- Skills score

The system trains multiple regression models, automatically selects the best performer, and deploys it through an interactive web interface.

---

## 🧠 Key Features

✅ Automated ML training pipeline  
✅ Feature engineering & preprocessing  
✅ Multiple regression model comparison  
✅ Automatic best model selection  
✅ Explainable AI using SHAP  
✅ Streamlit interactive web application  
✅ Modular and production-style project structure  

---

## 📂 Project Structure
salary_prediction/
│
├── app/
│ └── streamlit_app.py # Web application UI
│
├── data/
│ └── ds_salaries.csv # Dataset
│
├── models/
│ ├── Linear.pkl
│ ├── Ridge.pkl
│ ├── RandomForest.pkl
│ ├── XGBoost.pkl
│ └── best_model.pkl # Automatically selected best model
│
├── src/
│ ├── data_adapter.py # Dataset transformation layer
│ ├── feature_engineering.py # Feature creation logic
│ ├── preprocessing.py # ML preprocessing pipeline
│ ├── train.py # Model training pipeline
│ ├── evaluate.py # Model evaluation
│ ├── explain.py # SHAP explainability
│ └── predict.py # Prediction engine
│
├── requirements.txt
└── README.md

---

## 📊 Dataset

Dataset used:
**Data Science Salaries Dataset (Kaggle)**

The dataset contains real-world salary information including:

- Work year
- Experience level
- Employment type
- Job title
- Company size
- Company location
- Remote work ratio
- Salary in USD

---

## ⚙️ Implementation Details

---

### 1️⃣ Data Adapter (`data_adapter.py`)

Raw datasets rarely match ML model requirements.

This module:

- Loads the dataset
- Selects relevant columns
- Renames fields for consistency
- Converts categorical experience levels into numeric experience
- Simplifies job roles and locations
- Generates derived features like `SkillsScore`

This creates a clean dataset ready for ML processing.

---

### 2️⃣ Feature Engineering (`feature_engineering.py`)

Additional predictive features are created:

- `ExperienceSquared`
- `SkillExperienceInteraction`

These help models capture nonlinear salary growth patterns.

Feature engineering ensures better learning compared to raw inputs.

---

### 3️⃣ Preprocessing Pipeline (`preprocessing.py`)

Uses Scikit-Learn's `ColumnTransformer`:

- Numerical features → StandardScaler
- Categorical features → OneHotEncoder

This ensures:

- Proper scaling
- Model compatibility
- Automatic preprocessing during prediction

---

### 4️⃣ Model Training (`train.py`)

The training pipeline performs:

1. Dataset loading
2. Feature engineering
3. Data preprocessing
4. Train/Test split
5. Training multiple models:

   - Linear Regression
   - Ridge Regression
   - Random Forest Regressor
   - XGBoost Regressor

Each model is evaluated using **R² Score**.

---

### ⭐ Automatic Best Model Selection

After training:

```python
best_model = max(results, key=results.get)

The highest-performing model is saved as:

models/best_model.pkl

This mimics real production ML workflows.

5️⃣ Evaluation (evaluate.py)

Evaluates trained models using:

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

R² Score

This ensures model reliability.

6️⃣ Explainable AI (explain.py)

SHAP (SHapley Additive exPlanations) is used to interpret model behavior.

It explains:

Which features influence salary predictions

Global feature importance

This removes the "black box" nature of ML models.

7️⃣ Prediction Engine (predict.py)

Loads best_model.pkl and performs:

Input conversion to DataFrame

Feature engineering

Pipeline preprocessing

Salary prediction

Ensures consistency between training and inference.

8️⃣ Streamlit Web Application (streamlit_app.py)

Provides a user-friendly interface where users can:

Enter employee details

Predict salary instantly

Interact with the AI model

The app integrates directly with the trained pipeline.

Machine Learning Workflow
Dataset
   ↓
Data Adapter
   ↓
Feature Engineering
   ↓
Preprocessing Pipeline
   ↓
Model Training
   ↓
Best Model Selection
   ↓
Prediction API
   ↓
Streamlit Web App



📈 Model Performance

Models are compared using R² Score.

Typical results:

Model	R² Score
Linear Regression	~0.32
Ridge Regression	~0.33
Random Forest	~0.31
XGBoost	~0.30

Salary prediction is inherently noisy, making moderate R² values realistic.


▶️ How to Run the Project
1. Install dependencies
pip install -r requirements.txt
2. Train models
python src/train.py
3. Run explainability
python src/explain.py
4. Launch web application
streamlit run app/streamlit_app.py
🧩 Technologies Used

Python

Scikit-Learn

XGBoost

SHAP

Streamlit

Pandas & NumPy

🎯 Learning Outcomes

This project demonstrates:

End-to-end ML system design

Feature engineering techniques

Model comparison strategies

Explainable AI integration

Deployment of ML models into applications

🔮 Future Improvements

Local SHAP explanations per prediction

API deployment using FastAPI

Cloud deployment (Streamlit Cloud / AWS)

Real-time salary benchmarking dashboard

👨‍💻 Author

AI Salary Prediction System — Machine Learning Project

Built as an end-to-end regression-based AI application demonstrating practical ML engineering and deployment workflows.


