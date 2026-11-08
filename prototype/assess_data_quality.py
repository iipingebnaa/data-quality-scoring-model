"""
Research prototype: Predictive Data Quality Scoring Model
Stage 2: Assess completeness, accuracy, consistency and timeliness.

Validity and uniqueness are reported as supplementary checks.
The validation rules are prototype assumptions for synthetic data.
They are not official banking standards.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_DIR / "data" / "synthetic_onboarding_raw.csv"
OUTPUT_FILE = PROJECT_DIR / "data" / "onboarding_quality_assessed.csv"

REFERENCE_DATE = pd.Timestamp("2026-10-27")

df = pd.read_csv(INPUT_FILE, keep_default_na=False)

# Remove accidental spaces from text fields.
text_fields = [
    "full_name",
    "date_of_birth",
    "national_id_ref",
    "phone_number",
    "residential_address",
    "occupation",
]

for field in text_fields:
    df[field] = df[field].astype(str).str.strip()

# --------------------------------------------------
# 2. COMPLETENESS
# Check whether required fields have been provided.
# --------------------------------------------------

required_fields = [
    "full_name",
    "date_of_birth",
    "national_id_ref",
    "phone_number",
    "residential_address",
    "occupation",
]

completeness_checks = pd.DataFrame({
    field: df[field].ne("") for field in required_fields
})

df["completeness_score"] = (
    completeness_checks.mean(axis=1) * 100
).round(2)

# --------------------------------------------------
# 3. VALIDITY (SUPPLEMENTARY)
# Check whether supplied values follow expected
# formats or data types.
# Missing values are handled under completeness.
# --------------------------------------------------

dob = pd.to_datetime(
    df["date_of_birth"],
    errors="coerce"
)

declared_age = pd.to_numeric(
    df["declared_age"],
    errors="coerce"
)

income = pd.to_numeric(
    df["monthly_income"],
    errors="coerce"
)

duration = pd.to_numeric(
    df["session_duration_seconds"],
    errors="coerce"
)

validity_checks = pd.DataFrame({
    "dob_valid": (
        df["date_of_birth"].ne("")
        & dob.notna()
        & (dob <= REFERENCE_DATE)
    ),
    "declared_age_valid": (
        df["declared_age"].ne("")
        & declared_age.notna()
        & declared_age.between(18, 120)
    ),
    "income_valid": (
        df["monthly_income"].ne("")
        & income.notna()
        & (income >= 0)
    ),
    "duration_valid": (
        df["session_duration_seconds"].ne("")
        & duration.notna()
        & duration.between(1, 2400)
    ),
})

# Only assess supplied values.
validity_applicable = pd.DataFrame({
    "dob_parseable": df["date_of_birth"].ne(""),
    "declared_age_numeric": df["declared_age"].ne(""),
    "income_numeric": df["monthly_income"].ne(""),
    "duration_numeric": df["session_duration_seconds"].ne(""),
})

validity_denominator = validity_applicable.sum(axis=1)

df["validity_score"] = np.where(
    validity_denominator > 0,
    (
        validity_checks.sum(axis=1)
        / validity_denominator.clip(lower=1)
    ) * 100,
    np.nan,
).round(2)

# Distinguish missing, malformed, future and valid DOB values.
df["dob_status"] = np.select(
    [
        df["date_of_birth"].eq(""),
        dob.isna(),
        dob > REFERENCE_DATE,
    ],
    [
        "Missing",
        "Invalid format or date",
        "Future date",
    ],
    default="Valid",
)

# --------------------------------------------------
# 4. ACCURACY
# Apply format and plausibility checks to values
# that have actually been supplied.
# Missing values are handled under completeness.
# --------------------------------------------------

phone_valid = df["phone_number"].str.fullmatch(
    r"\+264\d{9}", na=False
)

id_valid = df["national_id_ref"].str.fullmatch(
    r"SIM-ID-\d{6}", na=False
)

dob_not_future = (
    dob.notna()
    & (dob <= REFERENCE_DATE)
)

income_nonnegative = (
    income.notna()
    & (income >= 0)
)

accuracy_checks = pd.DataFrame({
    "phone_valid": phone_valid,
    "id_valid": id_valid,
    "dob_not_future": dob_not_future,
    "income_nonnegative": income_nonnegative,
})

accuracy_applicable = pd.DataFrame({
    "phone_valid": df["phone_number"].ne(""),
    "id_valid": df["national_id_ref"].ne(""),
    "dob_not_future": df["date_of_birth"].ne(""),
    "income_nonnegative": df["monthly_income"].ne(""),
})

accuracy_denominator = accuracy_applicable.sum(axis=1)

df["accuracy_score"] = np.where(
    accuracy_denominator > 0,
    (
        (
            accuracy_checks & accuracy_applicable
        ).sum(axis=1)
        / accuracy_denominator.clip(lower=1)
    ) * 100,
    np.nan,
).round(2)

# --------------------------------------------------
# 5. CONSISTENCY
# Compare declared age with age calculated from DOB.
# --------------------------------------------------

calculated_age = (
    REFERENCE_DATE.year
    - dob.dt.year
    - (
        (REFERENCE_DATE.month < dob.dt.month)
        | (
            (REFERENCE_DATE.month == dob.dt.month)
            & (REFERENCE_DATE.day < dob.dt.day)
        )
    ).fillna(False).astype(int)
)

df["calculated_age"] = calculated_age

df["consistency_score"] = np.where(
    dob.notna()
    & (dob <= REFERENCE_DATE)
    & declared_age.notna()
    & (declared_age == calculated_age),
    100.0,
    0.0,
)

# --------------------------------------------------
# 6. TIMELINESS
# Prototype rule: session duration must be 15 minutes
# (900 seconds) or less.
# --------------------------------------------------

df["timeliness_score"] = np.where(
    duration.between(1, 900),
    100.0,
    0.0,
)

# --------------------------------------------------
# 7. UNIQUENESS (SUPPLEMENTARY)
# Check duplicate record IDs and identification refs.
# Empty identification references are not treated
# as unique.
# --------------------------------------------------

df["record_id_unique"] = (
    df["record_id"].ne("")
    & ~df["record_id"].duplicated(keep=False)
)

df["national_id_unique"] = (
    df["national_id_ref"].ne("")
    & ~df["national_id_ref"].duplicated(keep=False)
)

df["uniqueness_flag"] = np.where(
    df["record_id_unique"] & df["national_id_unique"],
    "Unique",
    "Review",
)

# --------------------------------------------------
# 8. COMPOSITE DATA-QUALITY SCORE
# Preserve the original four-dimension framework.
# Equal weighting is an initial baseline.
# --------------------------------------------------

score_columns = [
    "completeness_score",
    "accuracy_score",
    "consistency_score",
    "timeliness_score",
]

df["composite_quality_score"] = (
    df[score_columns].mean(axis=1)
).round(2)

# --------------------------------------------------
# 9. FLAG RECORDS FOR REVIEW
# This is a prototype threshold, not a validated
# banking acceptance threshold.
# --------------------------------------------------

df["quality_review_flag"] = np.where(
    df["composite_quality_score"] < 100,
    "Review",
    "Pass",
)

# --------------------------------------------------
# 10. SAVE ASSESSED DATASET
# --------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)

# --------------------------------------------------
# 11. DISPLAY SUMMARY RESULTS
# --------------------------------------------------

print("DATA-QUALITY ASSESSMENT COMPLETED")
print("=" * 50)

print(f"Records assessed: {len(df):,}")
print(f"Output file: {OUTPUT_FILE}")

print("\nAverage score by core dimension:")

for column in score_columns + ["composite_quality_score"]:
    print(f"{column}: {df[column].mean():.2f}%")

print("\nSupplementary validity results:")
print(f"Average validity score: {df['validity_score'].mean():.2f}%")
print(df["dob_status"].value_counts().to_string())

print("\nSupplementary uniqueness results:")
print(df["uniqueness_flag"].value_counts().to_string())

print("\nReview flag counts:")
print(df["quality_review_flag"].value_counts().to_string())

print("\nFirst five assessed records:")
print(
    df[
        [
            "record_id",
            "completeness_score",
            "accuracy_score",
            "consistency_score",
            "timeliness_score",
            "validity_score",
            "dob_status",
            "uniqueness_flag",
            "composite_quality_score",
            "quality_review_flag",
        ]
    ].head().to_string(index=False)
)