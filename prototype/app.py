
from pathlib import Path
from datetime import date
import os
import re

import joblib
import numpy as np
import pandas as pd

from flask import (
    Flask,
    render_template,
    render_template_string,
    request,
    redirect,
    url_for,
    session,
)

# --------------------------------------------------
# 1. APPLICATION SETUP
# --------------------------------------------------

app = Flask(__name__)

app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "local-prototype-development-key"
)

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_FILE = (
    PROJECT_DIR
    / "prototype"
    / "models"
    / "random_forest_combined.joblib"
)

REFERENCE_DATE = pd.Timestamp("2026-10-27")

model = joblib.load(MODEL_FILE)


# --------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------

def safe_number(value, default=0):

    try:
        number = float(value)

        if not np.isfinite(number):
            return default

        return number

    except (TypeError, ValueError):
        return default


def calculate_age(dob):

    if pd.isna(dob):
        return None

    return (
        REFERENCE_DATE.year
        - dob.year
        - (
            (REFERENCE_DATE.month, REFERENCE_DATE.day)
            < (dob.month, dob.day)
        )
    )


# --------------------------------------------------
# 3. CREATE MODEL FEATURES
# --------------------------------------------------

def create_model_features(data):

    full_name = data.get("full_name", "").strip()
    dob_text = data.get("date_of_birth", "").strip()
    national_id = data.get("national_id_ref", "").strip()
    phone = data.get("phone", "").strip()
    address = data.get("address", "").strip()
    occupation = data.get("occupation", "").strip()

    try:
        dob = pd.to_datetime(dob_text, errors="coerce")
    except (TypeError, ValueError):
        dob = pd.NaT

    declared_age = safe_number(data.get("declared_age"))
    monthly_income = safe_number(data.get("monthly_income"))

    session_duration = max(
        0,
        safe_number(data.get("session_duration_seconds"))
    )

    # Map website field names to training features.

    dob_valid = int(
        pd.notna(dob) and dob <= REFERENCE_DATE
    )

    calculated_age = calculate_age(dob)

    if calculated_age is not None:
        age_difference = abs(
            declared_age - calculated_age
        )
    else:
        age_difference = 0

    features = {

        "onboarding_channel":
            data.get("onboarding_channel", "Digital"),

        "declared_age":
            declared_age,

        "monthly_income":
            monthly_income,

        "session_duration_seconds":
            session_duration,

        "full_name_present":
            int(bool(full_name)),

        "dob_present":
            int(bool(dob_text)),

        "national_id_present":
            int(bool(national_id)),

        "phone_present":
            int(bool(phone)),

        "address_present":
            int(bool(address)),

        "occupation_present":
            int(bool(occupation)),

        "phone_format_valid":
            int(bool(re.fullmatch(r"\+264\d{9}", phone))),

        "national_id_format_valid":
            int(bool(re.fullmatch(r"SIM-ID-\d{6}", national_id))),

        "dob_valid":
            dob_valid,

        "income_nonnegative":
            int(monthly_income >= 0),

        "age_difference":
            age_difference,

        "avg_typing_cadence":
            max(0, safe_number(data.get("avg_typing_cadence"))),

        "avg_flight_time_ms":
            max(0, safe_number(data.get("avg_flight_time_ms"))),

        "avg_input_latency_ms":
            max(0, safe_number(data.get("avg_input_latency_ms"))),

        "avg_dwell_time_ms":
            max(0, safe_number(data.get("avg_dwell_time_ms"))),

        "correction_count":
            max(0, safe_number(data.get("correction_count"))),
    }

    return features, dob


# --------------------------------------------------
# 4. DATA QUALITY ASSESSMENT
# --------------------------------------------------

def assess_data_quality(data, dob):

    full_name = data.get("full_name", "").strip()
    dob_text = data.get("date_of_birth", "").strip()
    national_id = data.get("national_id_ref", "").strip()
    phone = data.get("phone", "").strip()
    address = data.get("address", "").strip()
    occupation = data.get("occupation", "").strip()

    age = safe_number(data.get("declared_age"))
    income = safe_number(data.get("monthly_income"))

    duration = safe_number(
        data.get("session_duration_seconds")
    )

    # COMPLETENESS

    required_values = [
        full_name,
        dob_text,
        national_id,
        phone,
        address,
        occupation,
    ]

    completeness = (
        sum(bool(value) for value in required_values)
        / len(required_values)
    ) * 100

    # VALIDITY (SUPPLEMENTARY)

    validity_checks = [
        pd.notna(dob) and dob <= REFERENCE_DATE,
        18 <= age <= 120,
        income >= 0,
        1 <= duration <= 2400,
    ]

    validity = (
        sum(validity_checks) / len(validity_checks)
    ) * 100

    # ACCURACY

    accuracy_checks = []

    if phone:
        accuracy_checks.append(
            bool(re.fullmatch(r"\+264\d{9}", phone))
        )

    if national_id:
        accuracy_checks.append(
            bool(re.fullmatch(r"SIM-ID-\d{6}", national_id))
        )

    if dob_text:
        accuracy_checks.append(
            pd.notna(dob) and dob <= REFERENCE_DATE
        )

    if data.get("monthly_income", "").strip():
        accuracy_checks.append(income >= 0)

    accuracy = (
        sum(accuracy_checks) / len(accuracy_checks) * 100
        if accuracy_checks else 0
    )

    # CONSISTENCY

    calculated_age = calculate_age(dob)

    consistency = (
        100.0
        if (
            calculated_age is not None
            and dob <= REFERENCE_DATE
            and age == calculated_age
        )
        else 0.0
    )

    # TIMELINESS

    timeliness = (
        100.0 if 1 <= duration <= 900 else 0.0
    )

    # COMPOSITE SCORE
    # Equal weighting of the four core dimensions.

    composite = np.mean([
        completeness,
        accuracy,
        consistency,
        timeliness,
    ])

    return {
        "completeness": round(completeness, 2),
        "accuracy": round(accuracy, 2),
        "consistency": round(consistency, 2),
        "timeliness": round(timeliness, 2),
        "validity": round(validity, 2),
        "composite": round(composite, 2),
        "review_flag": (
            "Review" if composite < 100 else "Pass"
        ),
        "uniqueness": (
            "Not assessed — single-record demonstration"
        ),
    }


# --------------------------------------------------
# 5. CUSTOMER ONBOARDING PAGE
# --------------------------------------------------

@app.route("/")
def home():

    return render_template("onboarding.html")


# --------------------------------------------------
# 6. PROCESS ONBOARDING SUBMISSION
# --------------------------------------------------

@app.route("/submit", methods=["POST"])
def submit():

    data = request.form.to_dict()

    features, dob = create_model_features(data)

    # Preserve the exact feature order expected by the model.

    model_input = pd.DataFrame(
        [features],
        columns=model.feature_names_in_
    )
    
    prediction = int(model.predict(model_input)[0])

    probability = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(model_input)[0]

        class_positions = list(model.classes_)

        if 1 in class_positions:
            probability = round(
                float(probabilities[class_positions.index(1)]) * 100,
                2
            )

    assessment = assess_data_quality(data, dob)

    result = {
        "assessment": assessment,
        "prediction": (
            "Potential quality issue detected"
            if prediction == 1
            else "No quality issue predicted"
        ),
        "prediction_probability": probability,
    }

    # Keep the latest demonstration result in this browser session.
    # No submitted personal details are stored.

    session["latest_result"] = result

    return render_template_string("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Submission Received</title>
            <link rel="stylesheet"
                  href="{{ url_for('static', filename='css/style.css') }}">
        </head>
        <body>
            <main class="page-container">
                <section class="form-card">
                    <h1>Submission received</h1>
                    <p>Your demonstration submission has been processed.</p>
                    <p>For privacy, this prototype does not retain the
                       personal details entered in the form.</p>
                    <p>The staff dashboard can display the assessment
                       after staff authentication.</p>
                    <a href="{{ url_for('home') }}">Return to onboarding</a>
                </section>
            </main>
        </body>
        </html>
    """)


# --------------------------------------------------
# 7. STAFF DASHBOARD LOGIN
# --------------------------------------------------

@app.route("/staff-login", methods=["GET", "POST"])
def staff_login():

    dashboard_password = os.environ.get("DASHBOARD_PASSWORD")

    if not dashboard_password:
        return (
            "Dashboard password is not configured. "
            "Set the DASHBOARD_PASSWORD environment variable "
            "before accessing the staff dashboard.",
            503
        )

    if request.method == "POST":

        entered_password = request.form.get("password", "")

        if entered_password == dashboard_password:

            session["staff_authenticated"] = True

            return redirect(url_for("dashboard"))

        return render_template_string("""
            <h2>Staff Dashboard Login</h2>
            <p>Incorrect password. Please try again.</p>
            <form method="POST">
                <input type="password" name="password"
                       placeholder="Staff password" required>
                <button type="submit">Sign in</button>
            </form>
        """)

    return render_template_string("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Staff Login</title>
            <link rel="stylesheet"
                  href="{{ url_for('static', filename='css/style.css') }}">
        </head>
        <body>
            <main class="page-container">
                <section class="form-card">
                    <h1>Staff Dashboard</h1>
                    <p>Authorised access only.</p>
                    <form method="POST">
                        <div class="form-group">
                            <label for="password">Dashboard Password</label>
                            <input type="password" id="password"
                                   name="password" required>
                        </div>
                        <button type="submit">Sign in</button>
                    </form>
                </section>
            </main>
        </body>
        </html>
    """)


# --------------------------------------------------
# 8. RESTRICTED STAFF DASHBOARD
# --------------------------------------------------

@app.route("/dashboard")
def dashboard():

    if not session.get("staff_authenticated"):
        return redirect(url_for("staff_login"))

    result = session.get("latest_result")

    if result is None:
        return (
            "No demonstration submission is available "
            "in this browser session yet.",
            404
        )

    return render_template(
        "dashboard.html",
        result=result
    )


# --------------------------------------------------
# 9. LOG OUT
# --------------------------------------------------

@app.route("/logout")
def logout():

    session.pop("staff_authenticated", None)

    return redirect(url_for("home"))


# --------------------------------------------------
# 10. RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
