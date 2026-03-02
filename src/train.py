import os
import joblib

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from data_adapter import load_and_prepare_data
from preprocessing import preprocess_data
from feature_engineering import create_features


def train_models(X, y, preprocessor):

    os.makedirs("models", exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            objective="reg:squarederror"
        )
    }

    results = {}

    for name, model in models.items():

        print(f"\nTraining {name}...")

        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        score = pipeline.score(X_test, y_test)
        results[name] = score

        joblib.dump(pipeline, f"models/{name}.pkl")

        print(f"{name} R2 Score: {score:.4f}")

    best_model_name = max(results, key=results.get)
    best_score = results[best_model_name]

    print(f"\n🏆 Best Model: {best_model_name} ({best_score:.4f})")

    best_model = joblib.load(f"models/{best_model_name}.pkl")
    joblib.dump(best_model, "models/best_model.pkl")

    print("Best model saved as models/best_model.pkl")

    return results


if __name__ == "__main__":

    print("\nStarting Salary Prediction Training Pipeline...\n")

    print("Loading dataset...")
    df = load_and_prepare_data()

    print("Creating features...")
    df = create_features(df)

    print("Preprocessing data...")
    X, y, preprocessor = preprocess_data(df)

    print("Training models...")
    results = train_models(X, y, preprocessor)

    print("\nTraining Complete!")
    print("Model Performance (R2 Scores):")

    for model, score in results.items():
        print(f"{model}: {score:.4f}")