# Predictive Data Quality Scoring Model for Customer Onboarding Data in the Banking Sector

## 1. Research Project

This repository contains the implementation of an Honours research project focused on developing a predictive data quality scoring model for customer onboarding data in the Namibian banking sector.

The project explores the assessment of customer onboarding data using four core data quality dimensions:

- Completeness
- Accuracy
- Consistency
- Timeliness

Machine learning techniques are also explored to identify potential data quality issues and support the assessment process.

## 2. Project Structure

```text
Data_Quality_Scoring_Model/
│
├── data/                       # Synthetic onboarding datasets
│
├── documentation/              # Technical and research documentation
│
├── notebooks/                  # Jupyter notebooks and experiments
│
├── prototype/
│   ├── models/                 # Trained machine learning models
│   ├── static/                 # CSS and JavaScript files
│   ├── templates/              # HTML interface templates
│   └── *.py                    # Python implementation scripts
│
├── research_documents/         # Research proposal and thesis documents
│
├── results/                    # Model evaluation results
│
├── README.md
└── .gitignore
```

## 3. Technologies Used

- Python 3.12
- Pandas
- NumPy
- Scikit-learn
- Flask
- Matplotlib
- Seaborn
- Joblib
- Visual Studio Code

## 4. Prototype Overview

The prototype consists of two main components:

**Customer Onboarding Interface**

A simulated onboarding form that accepts fictional customer information and captures selected numerical interaction measurements for research purposes.

**Data Quality Assessment Dashboard**

A restricted staff-facing dashboard that displays data quality scores, supplementary checks and machine learning predictions.

The predictive component uses trained Random Forest classification models to explore the identification of potential data quality issues.

## 5. Running the Prototype

The prototype is developed and tested locally using Python and Flask.

### Prerequisites

- Python 3.12 or later
- Visual Studio Code
- Required Python libraries

### Environment Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required libraries:

```powershell
pip install pandas numpy scikit-learn flask joblib matplotlib seaborn
```

### Start the Application

Configure the staff dashboard password in PowerShell:

```powershell
$env:DASHBOARD_PASSWORD = "YourOwnDemoPassword"
```

Start the Flask application:

```powershell
python prototype/app.py
```

### Accessing the Prototype

- Customer onboarding interface: http://127.0.0.1:5000/
- Staff login: http://127.0.0.1:5000/staff-login

The Flask application must remain running while using the prototype.

## 6. Data and Model Evaluation

The initial prototype uses 1,000 synthetically generated customer onboarding records.

The repository includes datasets for:

- Raw synthetic onboarding data.
- Data quality assessment results.
- Prepared onboarding features.
- Behavioural features.
- Combined modelling features.

Logistic Regression and Random Forest classification models are explored and evaluated. Their performance results are stored in the `results/` directory.

## 7. Data Privacy and Ethical Considerations

The prototype currently uses synthetic data. No actual customer records, personal banking information or confidential banking information are used.

The application is a research demonstration and is not a production banking system.

Any future collection of information from human participants will be subject to the relevant institutional ethical approval and consent requirements.

## 8. Research Limitations

The current evaluation is based on synthetic data and simulated behavioural measurements. The generated patterns and data quality rules represent prototype assumptions rather than established banking standards or observed customer behaviour.

Model performance on synthetic data should therefore not be interpreted as evidence of equivalent performance on real banking onboarding data.

Further validation would be required before any practical banking application.

## 9. Project Status

The repository contains the current research prototype, synthetic datasets, trained models and initial model evaluation results. Development and research documentation will be updated as the project progresses.