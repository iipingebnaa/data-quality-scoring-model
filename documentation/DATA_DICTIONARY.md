# Data Dictionary
## Predictive Data Quality Scoring Model for Customer Onboarding Data in the Banking Sector

### 1. Purpose

This document describes the variables contained in the five datasets developed for the research prototype. The datasets contain 1,000 synthetic customer onboarding records and are intended for controlled research and prototype evaluation.

No real customer information is included.

The datasets are:
1. `synthetic_onboarding_raw.csv`
2. `onboarding_quality_assessed.csv`
3. `onboarding_model_data.csv`
4. `behaviour_model_data.csv`
5. `combined_model_data.csv`

### 2. Raw synthetic onboarding dataset

**File:** `synthetic_onboarding_raw.csv`

This dataset contains the original generated onboarding records before the data quality assessment is applied.

| Variable | Data type | Description |
|---|---|---|
| record_id | String | Unique synthetic identifier assigned to each record. |
| onboarding_channel | String | Simulated onboarding channel, such as Assisted or Digital. |
| full_name | String | Fictional customer's full name. |
| date_of_birth | Date/string | Fictional customer's date of birth. |
| declared_age | Integer | Age declared during onboarding. |
| national_id_ref | String | Simulated national identification reference; not a real identity number. |
| phone_number | String | Synthetic contact telephone number. |
| residential_address | String | Fictional residential address. |
| occupation | String | Fictional customer's occupation. |
| monthly_income | Float | Synthetic monthly income value. |
| session_start | Datetime/string | Timestamp indicating the beginning of the simulated onboarding session. |
| submission_timestamp | Datetime/string | Timestamp indicating when the simulated submission was completed. |
| session_duration_seconds | Integer | Duration of the onboarding session in seconds. |
| avg_typing_cadence | Float | Average typing cadence measured in the simulated interaction. |
| avg_flight_time_ms | Float | Average time between successive key releases and presses, in milliseconds. |
| avg_input_latency_ms | Float | Average interval between recorded input events, in milliseconds. |
| avg_dwell_time_ms | Float | Average time a key remains pressed, in milliseconds. |
| correction_count | Integer | Number of recorded correction actions, such as deletion events. |
| injected_issue | String | Synthetic issue category assigned during data generation: None, Completeness, Accuracy, Consistency or Timeliness. |

**Important:** `injected_issue` is a controlled synthetic label used to construct and evaluate the prototype. It is not an issue classification obtained from actual bank records.

### 3. Assessed onboarding dataset

**File:** `onboarding_quality_assessed.csv`

This dataset contains the original onboarding variables together with the results of the data quality assessment.

| Variable | Data type | Description |
|---|---|---|
| record_id | String | Synthetic record identifier. |
| onboarding_channel | String | Simulated onboarding channel. |
| full_name | String | Fictional customer's full name. |
| date_of_birth | Date/string | Fictional customer's date of birth. |
| declared_age | Integer | Age declared by the customer. |
| national_id_ref | String | Simulated identification reference. |
| phone_number | String logically; numeric inference in CSV | Synthetic contact number. Should be treated as text because it is an identifier. |
| residential_address | String | Fictional residential address. |
| occupation | String | Fictional occupation. |
| monthly_income | Float | Synthetic monthly income. |
| session_start | Datetime/string | Start timestamp of the onboarding session. |
| submission_timestamp | Datetime/string | Submission timestamp. |
| session_duration_seconds | Integer | Duration of the session in seconds. |
| avg_typing_cadence | Float | Average simulated typing cadence. |
| avg_flight_time_ms | Float | Average flight time in milliseconds. |
| avg_input_latency_ms | Float | Average interval between input events in milliseconds. |
| avg_dwell_time_ms | Float | Average key dwell time in milliseconds. |
| correction_count | Integer | Number of recorded correction actions. |
| injected_issue | String | Controlled synthetic issue category. |
| completeness_score | Float | Assessment score for the presence of required information. |
| validity_score | Float | Assessment score for whether values meet the prototype's validity rules. |
| dob_status | String | Indicates whether the date of birth passes the date validation check. |
| accuracy_score | Float | Assessment score for selected value-format and plausibility checks. |
| calculated_age | Integer | Age calculated from the date of birth using the configured reference date. |
| consistency_score | Float | Assessment score for agreement between declared age and calculated age. |
| timeliness_score | Float | Assessment score based on the configured onboarding session-duration threshold. |
| record_id_unique | Boolean | Indicates whether the synthetic record identifier is unique. |
| national_id_unique | Boolean | Indicates whether the simulated identification reference is unique. |
| uniqueness_flag | String | Summary indicator of the uniqueness assessment. |
| composite_quality_score | Float | Equal-weight composite of completeness, accuracy, consistency and timeliness scores. |
| quality_review_flag | String | Overall prototype review indicator, such as Pass or Review. |

The composite score is calculated from four core dimensions:

- Completeness
- Accuracy
- Consistency
- Timeliness

Validity and uniqueness are supplementary indicators and are not included in the four-dimension composite score.

### 4. Onboarding model dataset

**File:** `onboarding_model_data.csv`

This dataset contains onboarding-related predictors prepared for classification modelling.

| Variable | Data type | Description |
|---|---|---|
| onboarding_channel | String | Simulated onboarding channel. |
| declared_age | Integer | Age declared during onboarding. |
| monthly_income | Float | Synthetic monthly income. |
| session_duration_seconds | Integer | Onboarding session duration. |
| full_name_present | Integer | Binary indicator showing whether the full name is present. |
| dob_present | Integer | Binary indicator showing whether date of birth is present. |
| national_id_present | Integer | Binary indicator showing whether the identification reference is present. |
| phone_present | Integer | Binary indicator showing whether a phone number is present. |
| address_present | Integer | Binary indicator showing whether an address is present. |
| occupation_present | Integer | Binary indicator showing whether occupation is present. |
| phone_format_valid | Integer | Binary indicator showing whether the phone number meets the configured format rule. |
| national_id_format_valid | Integer | Binary indicator showing whether the identification reference meets the configured format rule. |
| dob_valid | Integer | Binary indicator showing whether the date of birth passes the configured validity rule. |
| income_nonnegative | Integer | Binary indicator showing whether monthly income is non-negative. |
| age_difference | Integer | Difference between declared age and calculated age. |
| quality_issue | Integer | Classification target: 0 indicates no injected issue and 1 indicates an injected quality issue. |
| composite_quality_score | Float | Composite assessment score retained for analysis but excluded from model predictors. |

### 5. Behavioural model dataset

**File:** `behaviour_model_data.csv`

This dataset contains behavioural interaction measures used to investigate whether simulated interaction patterns can help distinguish records with and without injected quality issues.

| Variable | Data type | Description |
|---|---|---|
| avg_typing_cadence | Float | Average simulated typing cadence. |
| avg_flight_time_ms | Float | Average time between successive key release and press events. |
| avg_input_latency_ms | Float | Average interval between input events. |
| avg_dwell_time_ms | Float | Average key press duration. |
| correction_count | Integer | Number of recorded correction actions. |
| quality_issue | Integer | Classification target: 0 indicates no injected issue and 1 indicates an injected issue. |
| composite_quality_score | Float | Composite assessment score retained for analysis but excluded from model predictors. |

### 6. Combined model dataset

**File:** `combined_model_data.csv`

This dataset combines the onboarding and behavioural predictors.

It contains the following variables:

- `onboarding_channel`
- `declared_age`
- `monthly_income`
- `session_duration_seconds`
- `full_name_present`
- `dob_present`
- `national_id_present`
- `phone_present`
- `address_present`
- `occupation_present`
- `phone_format_valid`
- `national_id_format_valid`
- `dob_valid`
- `income_nonnegative`
- `age_difference`
- `avg_typing_cadence`
- `avg_flight_time_ms`
- `avg_input_latency_ms`
- `avg_dwell_time_ms`
- `correction_count`
- `quality_issue`
- `composite_quality_score`

The dataset contains 20 predictor variables, one classification target and one composite assessment score.

### 7. Variable roles in classification

The classification models use `quality_issue` as the target variable.

| Target value | Meaning |
|---|---|
| 0 | No injected quality issue |
| 1 | An injected quality issue is present |

Both training scripts exclude:
- `quality_issue`, because it is the target.
- `composite_quality_score`, because it is an assessment outcome rather than an independent predictor.

The remaining eligible columns are used as predictors.

### 8. Data preparation and interpretation notes

1. Binary presence and validation indicators use 1 to represent true/present/valid and 0 to represent false/absent/invalid.
2. Telephone numbers and identification references should be treated as strings, not numerical measurements.
3. Date and timestamp fields are stored as strings in the CSV files and should be parsed as dates or datetimes when required.
4. The datasets contain synthetic values and controlled issue patterns.
5. Behavioural measures are simulated in the generated dataset. Measurements captured by the live prototype are collected in a controlled demonstration environment and should not be treated as established customer behaviour.
6. The classification target is derived from the synthetic issue-generation label. Some predictors are closely related to the rules used to inject those issues. Consequently, model performance may be optimistic and should not be generalised to real banking data.
7. The configured assessment thresholds and validation rules are prototype assumptions, not official banking standards.

### 9. Intended use

The data dictionary supports the interpretation, reproducibility and documentation of the research prototype. It is intended to help the supervisor understand the structure of the datasets, the meaning of the variables and their roles in the data quality assessment and classification processes.