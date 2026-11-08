
import pandas as pd

# Load modelling dataset
df = pd.read_csv(
    "data/onboarding_model_data.csv",
    keep_default_na=False
)

print("MISSING VALUES BY COLUMN")
print("------------------------")

# Count empty values in each column
missing_counts = df.isin([""]).sum()

print(missing_counts[missing_counts > 0])

print("\nTOTAL EMPTY VALUES:")
print(missing_counts.sum())

print("\nMISSING VALUES BY TARGET CLASS:")
print(
    df.assign(missing_values=df.isin([""]).sum(axis=1))
      .groupby("quality_issue")["missing_values"]
      .sum()
)

print("\nDATASET PREVIEW:")
print(df.head().to_string())
