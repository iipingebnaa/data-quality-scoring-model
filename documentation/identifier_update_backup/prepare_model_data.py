
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. Define file locations
# --------------------------------------------------

DATA_DIR = Path("data")

INPUT_FILE = DATA_DIR / "onboarding_quality_assessed.csv"

ONBOARDING_OUTPUT = DATA_DIR / "onboarding_model_data.csv"
BEHAVIOUR_OUTPUT = DATA_DIR / "behaviour_model_data.csv"
COMBINED_OUTPUT = DATA_DIR / "combined_model_data.csv"

REFERENCE_DATE = pd.Timestamp("2026-10-27")

# --------------------------------------------------
# 2. Load the assessed dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE, keep_default_na=False)

# --------------------------------------------------
# 3. Create prediction targets
# --------------------------------------------------

# Classification target:
# 0 = no deliberately injected issue
# 1 = at least one deliberately injected issue

df["quality_issue"] = (
    df["injected_issue"] != "None"
).astype(int)

# Regression target:
# Continuous composite data-quality score (0-100)

df["composite_quality_score"] = pd.to_numeric(
    df["composite_quality_score"],
    errors="coerce"
)

# --------------------------------------------------
# 4. Engineer non-identifying onboarding features
# --------------------------------------------------

# Check whether required onboarding fields are populated.

required_fields = {
    "full_name": "full_name_present",
    "date_of_birth": "dob_present",
    "national_id_ref": "national_id_present",
    "phone_number": "phone_present",
    "residential_address": "address_present",
    "occupation": "occupation_present"
}

for source_column, new_column in required_fields.items():
    df[new_column] = (
        df[source_column].astype(str).str.strip() != ""
    ).astype(int)

# Check phone number format.

df["phone_format_valid"] = (
    df["phone_number"]
    .astype(str)
    .str.fullmatch(r"\+264\d{9}")
).astype(int)

# Check fictional national ID reference format.

df["national_id_format_valid"] = (
    df["national_id_ref"]
    .astype(str)
    .str.fullmatch(r"SIM-ID-\d{6}")
).astype(int)

# Check date of birth validity without retaining the raw DOB.

parsed_dob = pd.to_datetime(
    df["date_of_birth"],
    errors="coerce"
)

df["dob_valid"] = (
    parsed_dob.notna()
    & (parsed_dob <= REFERENCE_DATE)
).astype(int)

# Convert numeric fields safely.

df["declared_age"] = pd.to_numeric(
    df["declared_age"],
    errors="coerce"
)

df["monthly_income"] = pd.to_numeric(
    df["monthly_income"],
    errors="coerce"
)

df["session_duration_seconds"] = pd.to_numeric(
    df["session_duration_seconds"],
    errors="coerce"
)

# Check whether monthly income is non-negative.

df["income_nonnegative"] = (
    df["monthly_income"].notna()
    & (df["monthly_income"] >= 0)
).astype(int)

# Calculate the difference between declared age and
# age derived from date of birth.

# Determine whether the birthday has not yet occurred
# by the reference date.

birthday_not_reached = (
    (parsed_dob.dt.month > REFERENCE_DATE.month)
    |
    (
        (parsed_dob.dt.month == REFERENCE_DATE.month)
        & (parsed_dob.dt.day > REFERENCE_DATE.day)
    )
)

# Calculate age using the reference date.

calculated_age = (
    REFERENCE_DATE.year
    - parsed_dob.dt.year
    - birthday_not_reached.astype(int)
)

df["age_difference"] = (
    df["declared_age"] - calculated_age
).abs()

# Missing or invalid DOB values cannot provide an age difference.

df.loc[parsed_dob.isna(), "age_difference"] = 0

# --------------------------------------------------
# 5. Define onboarding and behavioural features
# --------------------------------------------------

onboarding_features = [
    "onboarding_channel",
    "declared_age",
    "monthly_income",
    "session_duration_seconds",
    "full_name_present",
    "dob_present",
    "national_id_present",
    "phone_present",
    "address_present",
    "occupation_present",
    "phone_format_valid",
    "national_id_format_valid",
    "dob_valid",
    "income_nonnegative",
    "age_difference"
]

behaviour_features = [
    "avg_typing_cadence",
    "avg_flight_time_ms",
    "avg_input_latency_ms",
    "avg_dwell_time_ms",
    "correction_count"
]

# Convert behavioural measurements to numeric values.

for column in behaviour_features:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# Define the prediction targets.

target_columns = [
    "quality_issue",
    "composite_quality_score"
]

# --------------------------------------------------
# 6. Create the three modelling datasets
# --------------------------------------------------

# Dataset 1: Onboarding information only.

onboarding_df = df[
    onboarding_features + target_columns
].copy()

# Dataset 2: Behavioural information only.

behaviour_df = df[
    behaviour_features + target_columns
].copy()

# Dataset 3: Onboarding and behavioural information.

combined_df = df[
    onboarding_features
    + behaviour_features
    + target_columns
].copy()

# --------------------------------------------------
# 7. Save the modelling datasets
# --------------------------------------------------

onboarding_df.to_csv(ONBOARDING_OUTPUT, index=False)
behaviour_df.to_csv(BEHAVIOUR_OUTPUT, index=False)
combined_df.to_csv(COMBINED_OUTPUT, index=False)

# --------------------------------------------------
# 8. Display verification results
# --------------------------------------------------

print("=" * 55)
print("MODELLING DATASETS PREPARED SUCCESSFULLY")
print("=" * 55)

print("\nDataset dimensions:")

print(
    f"Onboarding-only dataset: "
    f"{onboarding_df.shape}"
)

print(
    f"Behaviour-only dataset: "
    f"{behaviour_df.shape}"
)

print(
    f"Combined dataset: "
    f"{combined_df.shape}"
)

print("\nClassification target distribution:")

print(
    onboarding_df["quality_issue"]
    .value_counts()
    .sort_index()
)

print("\nClassification target proportions:")

print(
    onboarding_df["quality_issue"]
    .value_counts(normalize=True)
    .sort_index()
    .round(3)
)

print("\nComposite quality score summary:")

print(
    onboarding_df["composite_quality_score"]
    .describe()
    .round(2)
)

print("\nMissing values in each dataset:")

print(
    "Onboarding:",
    onboarding_df.isna().sum().sum()
)

print(
    "Behaviour:",
    behaviour_df.isna().sum().sum()
)

print(
    "Combined:",
    combined_df.isna().sum().sum()
)

print("\nSaved files:")

print(ONBOARDING_OUTPUT.resolve())
print(BEHAVIOUR_OUTPUT.resolve())
print(COMBINED_OUTPUT.resolve())

print("\n" + "=" * 55)
