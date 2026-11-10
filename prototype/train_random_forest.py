
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "prototype" / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Datasets to compare
datasets = {
    "Onboarding": DATA_DIR / "onboarding_model_data.csv",
    "Behaviour": DATA_DIR / "behaviour_model_data.csv",
    "Combined": DATA_DIR / "combined_model_data.csv"
}

results = []


for dataset_name, file_path in datasets.items():

    print(f"\n{'=' * 55}")
    print(f"Training Random Forest: {dataset_name}")
    print(f"{'=' * 55}")

    # Load dataset
    df = pd.read_csv(file_path)

    # Separate predictors and target
    X = df.drop(columns=["quality_issue", "composite_quality_score"])
    y = df["quality_issue"]

    # Identify categorical and numerical columns
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # Preprocessing
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Random Forest model
    classifier = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    # Complete modelling pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    matrix = confusion_matrix(y_test, y_pred)

    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save trained model
    model_path = MODEL_DIR / f"random_forest_{dataset_name.lower()}.joblib"
    joblib.dump(model, model_path)

    # Store results
    results.append({
        "Model": "Random Forest",
        "Dataset": dataset_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": roc_auc,
        "TN": matrix[0, 0],
        "FP": matrix[0, 1],
        "FN": matrix[1, 0],
        "TP": matrix[1, 1]
    })


# Save comparison results
results_df = pd.DataFrame(results)

output_file = RESULTS_DIR / "random_forest_results.csv"
results_df.to_csv(output_file, index=False)

print("\n" + "=" * 55)
print("RANDOM FOREST EXPERIMENT COMPLETED")
print("=" * 55)

print(results_df.to_string(index=False))
print(f"\nResults saved to: {output_file}")
