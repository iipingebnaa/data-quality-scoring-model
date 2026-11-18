from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "validation"
MODEL_DIR = PROJECT_DIR / "prototype" / "models"

DATASETS = {
    "Onboarding": (
        DATA_DIR / "independent_onboarding_model_data.csv",
        MODEL_DIR / "random_forest_onboarding.joblib",
    ),
    "Behaviour": (
        DATA_DIR / "independent_behaviour_model_data.csv",
        MODEL_DIR / "random_forest_behaviour.joblib",
    ),
    "Combined": (
        DATA_DIR / "independent_combined_model_data.csv",
        MODEL_DIR / "random_forest_combined.joblib",
    ),
}

results = []

# --------------------------------------------------
# 2. LOAD AND EVALUATE EXISTING MODELS
# --------------------------------------------------

for dataset_name, (data_file, model_file) in DATASETS.items():

    print("\n" + "=" * 60)
    print(f"{dataset_name.upper()} MODEL - INDEPENDENT VALIDATION")
    print("=" * 60)

    df = pd.read_csv(data_file)

    model = joblib.load(model_file)

    X = df.drop(
        columns=["quality_issue", "composite_quality_score"]
    )

    y = df["quality_issue"]

    # Ensure features match the original training order.
    X = X[model.feature_names_in_]

    # Predict using the existing trained model.
    predictions = model.predict(X)

    positive_class_index = list(model.classes_).index(1)

    probabilities = model.predict_proba(X)[
        :, positive_class_index
    ]

    # Calculate evaluation metrics.
    accuracy = accuracy_score(y, predictions)
    precision = precision_score(
        y, predictions, zero_division=0
    )
    recall = recall_score(
        y, predictions, zero_division=0
    )
    f1 = f1_score(
        y, predictions, zero_division=0
    )
    roc_auc = roc_auc_score(y, probabilities)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    matrix = confusion_matrix(
        y, predictions, labels=[0, 1]
    )

    tn, fp, fn, tp = matrix.ravel()

    print(f"Records evaluated: {len(df)}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(
        classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=["No Issue", "Quality Issue"],
            zero_division=0,
        )
    )

    results.append({
        "Model": "Random Forest",
        "Dataset": dataset_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": roc_auc,
        "True_Negatives": tn,
        "False_Positives": fp,
        "False_Negatives": fn,
        "True_Positives": tp,
    })

# --------------------------------------------------
# 3. SAVE INDEPENDENT VALIDATION RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

OUTPUT_FILE = (
    DATA_DIR / "independent_model_validation_results.csv"
)

results_df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 60)
print("INDEPENDENT MODEL VALIDATION COMPLETED")
print("=" * 60)

print(results_df.to_string(index=False))

print(f"\nResults saved to: {OUTPUT_FILE}")
