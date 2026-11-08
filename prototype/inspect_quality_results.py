
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_DIR / "data" / "onboarding_quality_assessed.csv"

df = pd.read_csv(INPUT_FILE)

print("QUALITY RESULTS DIAGNOSTIC")
print("=" * 40)

score_columns = [
    "completeness_score",
    "accuracy_score",
    "consistency_score",
    "timeliness_score",
]

for column in score_columns:
    failed = (df[column] < 100).sum()
    print(f"{column}: {failed} records below 100")

print("\nComposite score distribution:")
print(
    df["composite_quality_score"]
    .describe()
    .round(2)
    .to_string()
)

print("\nReview flag counts:")
print(df["quality_review_flag"].value_counts().to_string())

print("\nReview flag by onboarding channel:")
print(
    pd.crosstab(
        df["onboarding_channel"],
        df["quality_review_flag"],
        margins=True,
    ).to_string()
)
