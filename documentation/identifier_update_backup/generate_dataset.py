
"""
Research: A Predictive Data Quality Scoring Model for
Customer Onboarding Data in the Banking Sector.

Stage 1: Generate 1,000 synthetic onboarding records.
All records are fictional and intended for prototype evaluation.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. Reproducibility and project folders
# --------------------------------------------------

RANDOM_SEED = 42
N_RECORDS = 1000

rng = np.random.default_rng(RANDOM_SEED)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Reference date for calculating age and validating dates.
REFERENCE_DATE = pd.Timestamp("2026-10-27")

# --------------------------------------------------
# 2. Fictional information for the prototype
# --------------------------------------------------

first_names = [
    "Amutenya", "Ndeshi", "Tomas", "Selma", "Kandume",
    "Martha", "Petrus", "Nangula", "Johannes", "Tuli",
    "Ester", "Paulus", "Maria", "Ndapewa", "David",
    "Helvi", "Simon", "Lydia", "Samuel", "Monica"
]

last_names = [
    "Shilongo", "Amadhila", "Nghidinwa", "Uusiku",
    "Nambala", "Iiyambo", "Kandjii", "Nekongo",
    "Hamukoto", "Mupupa", "Kahuure", "Tjitemisa",
    "Nghipandulwa", "Ndeitunga", "Haingura"
]

occupations = [
    "Teacher", "Nurse", "Accountant", "Student",
    "Shop Assistant", "Driver", "Farmer", "Administrator",
    "Engineer", "Self-employed"
]

address_areas = [
    "Fictional Area A", "Fictional Area B",
    "Fictional Area C", "Fictional Area D",
    "Fictional Area E"
]

# --------------------------------------------------
# 3. Generate the initial clean dataset
# --------------------------------------------------

records = []

for i in range(1, N_RECORDS + 1):
    birth_year = int(rng.integers(1960, 2005))
    birth_month = int(rng.integers(1, 13))

    # Choose a valid day for the selected month.
    birth_day_max = (
        pd.Timestamp(birth_year, birth_month, 1)
        + pd.offsets.MonthEnd(0)
    ).day
    birth_day = int(rng.integers(1, birth_day_max + 1))

    dob = pd.Timestamp(birth_year, birth_month, birth_day)
    age = (
        REFERENCE_DATE.year - dob.year
        - (
            (REFERENCE_DATE.month, REFERENCE_DATE.day)
            < (dob.month, dob.day)
        )
    )

    start = REFERENCE_DATE - pd.Timedelta(
        days=int(rng.integers(0, 180))
    )
    start = start + pd.Timedelta(
        seconds=int(rng.integers(0, 86400))
    )

    duration = int(rng.integers(60, 901))
    submitted = start + pd.Timedelta(seconds=duration)

    records.append({
        "record_id": f"REC-{i:06d}",
        "onboarding_channel": rng.choice(
            ["Digital", "Assisted"],
            p=[0.60, 0.40]
        ),
        "full_name": (
            f"{rng.choice(first_names)} "
            f"{rng.choice(last_names)}"
        ),
        "date_of_birth": dob.strftime("%Y-%m-%d"),
        "declared_age": int(age),
        # Deliberately fictional reference, not a real national ID.
        "national_id_ref": f"SIM-ID-{i:06d}",
        # Reserved fictional test range; not for contacting anyone.
        "phone_number": (
            "+26481" + f"{int(rng.integers(0, 10000000)):07d}"
        ),
        "residential_address": (
            f"{int(rng.integers(1, 200))} "
            f"{rng.choice(address_areas)}"
        ),
        "occupation": rng.choice(occupations),
        "monthly_income": round(
            float(rng.uniform(2500, 45000)), 2
        ),
        "session_start": start.strftime("%Y-%m-%d %H:%M:%S"),
        "submission_timestamp": submitted.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "session_duration_seconds": duration,
        "avg_typing_cadence": round(
            float(rng.uniform(1.5, 8.0)), 2
        ),
        "avg_flight_time_ms": round(
            float(rng.uniform(80, 400)), 2
        ),
        "avg_input_latency_ms": round(
            float(rng.uniform(100, 1800)), 2
        ),
        "avg_dwell_time_ms": round(
            float(rng.uniform(200, 2500)), 2
        ),
        "correction_count": int(rng.integers(0, 16))
    })

df = pd.DataFrame(records)

# --------------------------------------------------
# 4. Introduce controlled data-quality issues
# --------------------------------------------------
# Each category is assigned to a different group of records.
# This gives us clearer labels for the first prototype.
# These are simulated errors, not observed banking errors.

record_indices = rng.permutation(N_RECORDS)

completeness_indices = record_indices[0:100]
accuracy_indices = record_indices[100:200]
consistency_indices = record_indices[200:300]
timeliness_indices = record_indices[300:400]

# Record the intended synthetic issue for each row.
# "None" means no issue was deliberately injected.
df["injected_issue"] = "None"

df.loc[completeness_indices, "injected_issue"] = "Completeness"
df.loc[accuracy_indices, "injected_issue"] = "Accuracy"
df.loc[consistency_indices, "injected_issue"] = "Consistency"
df.loc[timeliness_indices, "injected_issue"] = "Timeliness"

# 4a. Completeness: leave one required field blank.
required_fields_for_injection = [
    "full_name",
    "phone_number",
    "residential_address",
    "occupation"
]

for idx in completeness_indices:
    field = rng.choice(required_fields_for_injection)
    df.loc[idx, field] = ""

# 4b. Accuracy-related rule violations.
for idx in accuracy_indices:
    violation = rng.choice([
        "phone", "id", "dob", "income"
    ])

    if violation == "phone":
        df.loc[idx, "phone_number"] = "INVALID-PHONE"

    elif violation == "id":
        df.loc[idx, "national_id_ref"] = (
            f"INVALID-REFERENCE-{idx:06d}"
        )

    elif violation == "dob":
        df.loc[idx, "date_of_birth"] = "2035-01-01"

    else:
        df.loc[idx, "monthly_income"] = -500.00

# 4c. Consistency: declared age disagrees with date of birth.
for idx in consistency_indices:
    df.loc[idx, "declared_age"] = (
        int(df.loc[idx, "declared_age"]) + 10
    )

# 4d. Timeliness: session exceeds the prototype's 15-minute limit.
for idx in timeliness_indices:
    duration = int(rng.integers(901, 2401))
    start = pd.to_datetime(df.loc[idx, "session_start"])
    submitted = start + pd.Timedelta(seconds=duration)

    df.loc[idx, "session_duration_seconds"] = duration
    df.loc[idx, "submission_timestamp"] = (
        submitted.strftime("%Y-%m-%d %H:%M:%S")
    )

# --------------------------------------------------
# 5. Introduce controlled behavioural variation
# --------------------------------------------------
# Records with injected issues receive modest shifts in their
# behavioural measurements. Random variation and overlapping
# distributions are retained.
#
# These are simulation assumptions for prototype evaluation.
# They are NOT established behavioural patterns from real
# banking customers.

behaviour_rng = np.random.default_rng(RANDOM_SEED + 1)

issue_mask = df["injected_issue"] != "None"
number_of_issue_records = int(issue_mask.sum())

# Slightly lower typing cadence for records with injected issues.
df.loc[issue_mask, "avg_typing_cadence"] = (
    df.loc[issue_mask, "avg_typing_cadence"]
    - behaviour_rng.normal(
        0.25, 0.15, number_of_issue_records
    )
).clip(1.5, 8.0).round(2)

# Slightly higher average flight time.
df.loc[issue_mask, "avg_flight_time_ms"] = (
    df.loc[issue_mask, "avg_flight_time_ms"]
    + behaviour_rng.normal(
        15, 10, number_of_issue_records
    )
).clip(80, 400).round(2)

# Slightly higher average input latency.
df.loc[issue_mask, "avg_input_latency_ms"] = (
    df.loc[issue_mask, "avg_input_latency_ms"]
    + behaviour_rng.normal(
        120, 80, number_of_issue_records
    )
).clip(100, 1800).round(2)

# Slightly higher average dwell time.
df.loc[issue_mask, "avg_dwell_time_ms"] = (
    df.loc[issue_mask, "avg_dwell_time_ms"]
    + behaviour_rng.normal(
        100, 75, number_of_issue_records
    )
).clip(200, 2500).round(2)

# Slightly more corrections on average.
df.loc[issue_mask, "correction_count"] = (
    df.loc[issue_mask, "correction_count"]
    + behaviour_rng.normal(
        1.5, 1.0, number_of_issue_records
    )
).round().clip(0, 15).astype(int)

# --------------------------------------------------
# 6. Save the raw synthetic dataset
# --------------------------------------------------

raw_output = DATA_DIR / "synthetic_onboarding_raw.csv"
df.to_csv(raw_output, index=False)

print("Synthetic dataset created successfully.")
print(f"Number of records: {len(df):,}")
print(f"Number of columns: {len(df.columns)}")

print("\nInjected issue distribution:")
print(df["injected_issue"].value_counts())

print("\nSaved to:", raw_output)

print("\nFirst five records:")
print(df.head().to_string(index=False))
