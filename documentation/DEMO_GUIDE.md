# Demonstration Guide

## 1. Research Project

**Title:** A Predictive Data Quality Scoring Model for Customer Onboarding Data in the Banking Sector.

The project investigates how data quality assessment and machine learning can support the identification of potential data quality issues during customer onboarding.

## 2. Prototype Overview

The prototype consists of two main components:

- **Customer onboarding interface:** Simulates the submission of customer onboarding information and captures selected numerical interaction measures.
- **Staff dashboard:** Displays the data quality assessment, composite score, supplementary indicators and machine learning prediction. Access is password-protected.

The prototype uses 1,000 synthetic onboarding records. No real customer or confidential banking information is used.

## 3. Data Quality Assessment

The model assesses four main dimensions:

- Completeness
- Accuracy
- Consistency
- Timeliness

Validity and uniqueness are also considered as supplementary indicators.

The composite data quality score is calculated using the four main dimensions.

## 4. Model Development and Evaluation

Three Random Forest classification models were developed:

- Onboarding data model
- Behavioural data model
- Combined data model

The models were evaluated using a stratified training and testing split.

The combined and onboarding-only models achieved 100% accuracy on the synthetic test set, while the behavioural-only model achieved approximately 68.5%.

These results demonstrate performance within the constructed synthetic dataset. They should not be interpreted as evidence of equivalent performance on real banking data.

## 5. Prototype Demonstration Sequence

1. Open the customer onboarding interface.
2. Submit a fictional customer record.
3. Show the resulting data quality assessment.
4. Open the staff login page.
5. Log in to the restricted dashboard.
6. Explain the composite score and individual quality dimensions.
7. Show the machine learning prediction.
8. Demonstrate how missing or inconsistent information affects the assessment.
9. Explain the limitations and planned improvements.

## 6. Important Limitations

- The dataset is entirely synthetic.
- Behavioural measurements are simulated or captured in a controlled prototype environment.
- The model has not been validated using real banking onboarding data.
- High classification performance may partly reflect patterns deliberately introduced during synthetic data generation.
- The dashboard currently displays the latest submission in the browser session rather than maintaining a permanent submission history.
- The prototype is for research demonstration and is not a production banking application.

## 7. Ethics and Further Work

Formal ethics clearance from the University of Namibia is required before conducting research involving human participants or collecting participant data.

Further work includes refining the evaluation, documenting the implementation, completing the thesis chapters and conducting any approved participant testing.