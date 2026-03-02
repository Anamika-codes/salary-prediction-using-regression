import joblib
import pandas as pd
import shap

from data_adapter import load_and_prepare_data
from feature_engineering import create_features

# Load trained pipeline
model_pipeline = joblib.load("models/best_model.pkl")

# Extract preprocessing + trained model
preprocessor = model_pipeline.named_steps["preprocess"]
model = model_pipeline.named_steps["model"]

print("Loading dataset...")
df = load_and_prepare_data()

# Apply same feature engineering
df = create_features(df)

# Separate features
X = df.drop("Salary", axis=1)

# Transform data using preprocessing pipeline
X_transformed = preprocessor.transform(X)

print("Creating SHAP explainer...")

# Create SHAP explainer
explainer = shap.Explainer(model, X_transformed)

# Compute SHAP values (use sample for speed)
shap_values = explainer(X_transformed[:200])

print("Displaying feature importance plot...")

# Show global feature importance
shap.plots.bar(shap_values)