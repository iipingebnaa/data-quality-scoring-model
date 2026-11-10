
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# =========================================================
# 1. PROJECT PATHS AND SETTINGS
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET_COLUMN = "quality_issue"

DATASETS = {
    "Onboarding-only": "onboarding_model_data.csv",
    "Behaviour-only": "behaviour_model_data.csv",
    "Combined": "combined_model_data.csv"
}


# =========================================================
# 2. FUNCTION TO TRAIN AND EVALUATE LOGISTIC REGRESSION
# =========================================================

def train_logistic_regression(dataset_name, filename):

    print("\n" + "=" * 65)
    print(f"LOGISTIC REGRESSION: {dataset_name.upper()}")
    print("=" * 65)

    # Load dataset
    data = pd.read_csv(DATA_DIR / filename)

    # Separate predictors and classification target
    X = data.drop(
        columns=[TARGET_COLUMN, "composite_quality_score"]
    )

    y = data[TARGET_COLUMN]

    # Identify categorical and numerical columns
    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # Preprocess numerical features
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Preprocess categorical features
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine preprocessing
    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    # Build Logistic Regression model
    model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Train model
    model.fit(X_train, y_train)

    # Generate predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Display results
    print("\nEvaluation Metrics")
    print("-" * 40)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report")
    print(classification_report(
        y_test,
        y_pred,
        zero_division=0
    ))

    return {
        "Dataset": dataset_name,
        "Model": "Logistic Regression",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": roc_auc
    }


# =========================================================
# 3. TRAIN ON ALL THREE DATASETS
# =========================================================

def main():

    results = []

    for dataset_name, filename in DATASETS.items():

        result = train_logistic_regression(
            dataset_name,
            filename
        )

        results.append(result)

    # Create comparison table
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 65)
    print("LOGISTIC REGRESSION MODEL COMPARISON")
    print("=" * 65)

    print(results_df.to_string(index=False))

    # Save results
    output_path = RESULTS_DIR / "logistic_regression_results.csv"

    results_df.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
