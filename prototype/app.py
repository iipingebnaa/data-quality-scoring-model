
from pathlib import Path
from datetime import date, timedelta, timezone
import os
import re
import hmac
import hashlib
import uuid

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

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import SQLAlchemyError


# --------------------------------------------------
# 1. APPLICATION SETUP
# --------------------------------------------------

app = Flask(__name__)

app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "local-prototype-development-key"
)

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATABASE_DIR = PROJECT_DIR / "data"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

app.config["SQLALCHEMY_DATABASE_URI"] = (
    f"sqlite:///{DATABASE_DIR / 'onboarding.db'}"
)

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

MODEL_FILE = (
    PROJECT_DIR
    / "prototype"
    / "models"
    / "random_forest_combined.joblib"
)

REFERENCE_DATE = pd.Timestamp.today().normalize()

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
    document_type = data.get(
        "identification_document_type", ""
    ).strip()

    issuing_country = data.get(
        "issuing_country", ""
    ).strip()

    document_number = data.get(
        "identification_document_number", ""
    ).strip()
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
            int(bool(document_type and issuing_country and document_number)),

        "phone_present":
            int(bool(phone)),

        "address_present":
            int(bool(address)),

        "occupation_present":
            int(bool(occupation)),

        "phone_format_valid":
            int(bool(re.fullmatch(r"\+264\d{9}", phone))),

        "national_id_format_valid":
            int(bool(
              document_type in ["National ID", "Passport"]
              and issuing_country
              and document_number
            )),

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
    document_type = data.get(
    "identification_document_type", ""
      ).strip()

    issuing_country = data.get(
     "issuing_country", ""
      ).strip()

    document_number = data.get(
     "identification_document_number", ""
    ).strip()
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
        document_type,
        issuing_country,
        document_number,
        phone,
        address,
        occupation,
    ]

    completeness = (
        sum(bool(value) for value in required_values)
        / len(required_values)
    ) * 100

    # VALIDITY (SUPPLEMENTARY)

    
    # VALIDITY (SUPPLEMENTARY)

    validity_checks = [
        pd.notna(dob) and dob <= REFERENCE_DATE,
        18 <= age <= 120,
        income >= 0,
        1 <= duration <= 2400,
        bool(re.fullmatch(r"\+264\d{9}", phone)),
    ]

    validity = (
        sum(validity_checks) / len(validity_checks)
    ) * 100


    
    # ACCURACY

    accuracy_checks = []

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
# 5. DATABASE MODELS
# --------------------------------------------------

class Customer(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    customer_number = db.Column(
        db.String(30),
        unique=True,
        nullable=False
    )

    customer_match_key = db.Column(
        db.String(64),
        unique=True,
        nullable=True
    )


    identification_document_type = db.Column(
    db.String(30),
    nullable=True
    )

    issuing_country = db.Column(
    db.String(100),
    nullable=True
    )

    full_name = db.Column(db.String(150), nullable=True)

    date_of_birth = db.Column(db.String(20))

    phone = db.Column(db.String(30))

    address = db.Column(db.String(250))

    occupation = db.Column(db.String(100))

    monthly_income = db.Column(db.Float)

    created_at = db.Column(
        db.DateTime,
        server_default=db.func.now()
    )

    submissions = db.relationship(
        "Submission",
        back_populates="customer",
        cascade="all, delete-orphan"
    )


class Submission(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    submission_reference = db.Column(
        db.String(40),
        unique=True,
        nullable=False
    )

    customer_id = db.Column(
        db.Integer,
        db.ForeignKey("customer.id"),
        nullable=False
    )

    onboarding_channel = db.Column(db.String(30))

    completeness = db.Column(db.Float)
    accuracy = db.Column(db.Float)
    consistency = db.Column(db.Float)
    timeliness = db.Column(db.Float)
    validity = db.Column(db.Float)
    composite_score = db.Column(db.Float)

    review_flag = db.Column(db.String(30))
    uniqueness_status = db.Column(db.String(100))

    model_prediction = db.Column(db.String(50))
    prediction_probability = db.Column(db.Float)

    workflow_status = db.Column(
        db.String(30),
        default="Pending Review",
        nullable=False
    )

    created_at = db.Column(
        db.DateTime,
        server_default=db.func.now()
    )

    customer = db.relationship(
        "Customer",
        back_populates="submissions"
    )

    staff_review = db.relationship(
        "StaffReview",
        back_populates="submission",
        uselist=False,
        cascade="all, delete-orphan"
    )


class StaffReview(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    submission_id = db.Column(
        db.Integer,
        db.ForeignKey("submission.id"),
        unique=True,
        nullable=False
    )

    decision = db.Column(db.String(50))

    comments = db.Column(db.Text)

    reviewed_at = db.Column(
        db.DateTime,
        server_default=db.func.now()
    )

    submission = db.relationship(
        "Submission",
        back_populates="staff_review"
    )


# --------------------------------------------------
# 6. REFERENCE GENERATION
# --------------------------------------------------


def create_customer_match_key(
    document_type,
    issuing_country,
    document_number
):
    if not document_type or not issuing_country or not document_number:
       return None

    normalized_type = document_type.strip().upper()
    normalized_country = issuing_country.strip().upper()
    normalized_number = document_number.strip().upper()

    matching_value = (
        f"{normalized_type}|"
        f"{normalized_country}|"
        f"{normalized_number}"
    )

    return hmac.new(
        app.secret_key.encode("utf-8"),
        matching_value.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()


# --------------------------------------------------
# 7. CUSTOMER ONBOARDING PAGE
# --------------------------------------------------

@app.route("/")
def home():

    return render_template("onboarding.html")


# --------------------------------------------------
# 8. PROCESS AND SAVE ONBOARDING SUBMISSION
# --------------------------------------------------

@app.route("/submit", methods=["POST"])
def submit():

    data = request.form.to_dict()

    try:

        # Generate model features.

        features, dob = create_model_features(data)

        model_input = pd.DataFrame(
            [features],
            columns=model.feature_names_in_
        )

        # Generate model prediction.

        prediction = int(model.predict(model_input)[0])

        probability = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(model_input)[0]

            class_positions = list(model.classes_)

            if 1 in class_positions:
                probability = round(
                    float(
                        probabilities[class_positions.index(1)]
                    ) * 100,
                    2
                )

        # Assess data quality.

        assessment = assess_data_quality(data, dob)

        # Generate protected customer matching key.

        document_type = data.get(
            "identification_document_type", ""
        ).strip()

        issuing_country = data.get(
            "issuing_country", ""
        ).strip()

        document_number = data.get(
            "identification_document_number", ""
        ).strip()

        match_key = create_customer_match_key(
            document_type,
            issuing_country,
            document_number
        )

        # Find an existing customer where possible.

        customer = None

        if match_key:

            customer = Customer.query.filter_by(
                customer_match_key=match_key
            ).first()

        # Create a new customer if no match exists.

        if customer is None:

            customer = Customer(
                customer_number=f"TEMP-CUS-{uuid.uuid4().hex}",
                customer_match_key=match_key,
                identification_document_type=document_type or None,
                issuing_country=issuing_country or None,
                full_name=data.get("full_name", "").strip() or None,
                date_of_birth=data.get("date_of_birth", "").strip() or None,
                phone=data.get("phone", "").strip() or None,
                address=data.get("address", "").strip() or None,
                occupation=data.get("occupation", "").strip() or None,
                monthly_income=(
                    safe_number(data.get("monthly_income"))
                    if data.get("monthly_income", "").strip()
                    else None
                )
            )

            db.session.add(customer)

            db.session.flush()

            customer.customer_number = f"CUS-{1000 + customer.id}"

        else:

            # Update existing customer details when
            # new non-empty information is supplied.

            customer.identification_document_type = (
                document_type or customer.identification_document_type
            )

            customer.issuing_country = (
                issuing_country or customer.issuing_country
            )

            for field in [
                "full_name",
                "date_of_birth",
                "phone",
                "address",
                "occupation"
            ]:

                value = data.get(field, "").strip()

                if value:
                    setattr(customer, field, value)

            income = data.get("monthly_income", "").strip()

            if income:
                customer.monthly_income = safe_number(income)
            

        # Create an independent submission record.

        submission = Submission(

            submission_reference=f"TEMP-SUB-{uuid.uuid4().hex}",

            customer_id=customer.id,

            onboarding_channel=data.get("onboarding_channel"),

            completeness=assessment["completeness"],
            accuracy=assessment["accuracy"],
            consistency=assessment["consistency"],
            timeliness=assessment["timeliness"],
            validity=assessment["validity"],
            composite_score=assessment["composite"],

            review_flag=assessment["review_flag"],
            uniqueness_status=assessment["uniqueness"],

            model_prediction=(
                "Potential quality issue detected"
                if prediction == 1
                else "No quality issue predicted"
            ),

            prediction_probability=probability
        )

        db.session.add(submission)

        # Generate the sequential submission reference
        # using the database-assigned submission ID.

        db.session.flush()

        submission.submission_reference = f"SUB-{10000 + submission.id}"

        # Save customer and submission together.

        db.session.commit()

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

                        <p>Your demonstration submission
                           has been processed successfully.</p>

                        <p><strong>Customer number:</strong>
                           {{ customer_number }}</p>

                        <p><strong>Submission reference:</strong>
                           {{ submission_reference }}</p>

                        <p>Your submission has been recorded.
                           Authorised staff can review it
                           through the staff dashboard.</p>

                        <a href="{{ url_for('home') }}">
                           Return to onboarding
                        </a>

                    </section>
                </main>
            </body>
            </html>
        """,
            customer_number=customer.customer_number,
            submission_reference=submission.submission_reference
        )

    except SQLAlchemyError:

        db.session.rollback()

        app.logger.exception(
            "Database error while saving onboarding submission."
        )

        return (
            "The submission could not be saved. "
            "Please try again.",
            500
        )

    except Exception:

        db.session.rollback()

        app.logger.exception(
            "Unexpected error while processing onboarding submission."
        )

        return (
            "The submission could not be processed. "
            "Please contact the demonstration administrator.",
            500
        )


# --------------------------------------------------
# 9. STAFF DASHBOARD LOGIN
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

        if hmac.compare_digest(
            entered_password,
            dashboard_password
        ):

            session.clear()

            session["staff_authenticated"] = True

            return redirect(url_for("dashboard"))

        return render_template_string("""
            <h2>Staff Dashboard Login</h2>

            <p>Incorrect password. Please try again.</p>

            <form method="POST">
                <input type="password"
                       name="password"
                       placeholder="Staff password"
                       required>

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

                            <label for="password">
                                Dashboard Password
                            </label>

                            <input type="password"
                                   id="password"
                                   name="password"
                                   required>

                        </div>

                        <button type="submit">
                            Sign in
                        </button>

                    </form>

                </section>
            </main>
        </body>
        </html>
    """)


# --------------------------------------------------
# 10. RESTRICTED STAFF DASHBOARD
# --------------------------------------------------

@app.route("/dashboard")
def dashboard():

    if not session.get("staff_authenticated"):

        return redirect(url_for("staff_login"))

    # Retrieve the latest permanently stored submission.

    latest_submission = Submission.query.order_by(
        Submission.created_at.desc(),
        Submission.id.desc()
    ).first()

    if latest_submission is None:

        return (
            "No onboarding submissions have been recorded yet.",
            404
        )

    # Reconstruct the result structure expected by
    # the existing dashboard.html template.

    assessment = {

        "completeness": latest_submission.completeness,
        "accuracy": latest_submission.accuracy,
        "consistency": latest_submission.consistency,
        "timeliness": latest_submission.timeliness,
        "validity": latest_submission.validity,
        "composite": latest_submission.composite_score,

        "review_flag": latest_submission.review_flag,

        "uniqueness": latest_submission.uniqueness_status,
    }

    result = {

        "assessment": assessment,

        "prediction": latest_submission.model_prediction,

        "prediction_probability":
            latest_submission.prediction_probability,

        "customer_number":
            latest_submission.customer.customer_number,

        "submission_reference":
            latest_submission.submission_reference,
    }

    # Retrieve all submissions for the next dashboard update.

    submissions = Submission.query.order_by(
        Submission.created_at.desc(),
        Submission.id.desc()
    ).all()

    return render_template(
        "dashboard.html",
        result=result,
        submissions=submissions
    )


# --------------------------------------------------
# 11. LOG OUT
# --------------------------------------------------


# --------------------------------------------------
# 11. INDIVIDUAL SUBMISSION REVIEW
# --------------------------------------------------

@app.route("/review/<int:submission_id>", methods=["GET", "POST"])
def review_submission(submission_id):

    if not session.get("staff_authenticated"):
        return redirect(url_for("staff_login"))

    submission = db.session.get(Submission, submission_id)

    if submission is None:
        return "Submission not found.", 404

    if request.method == "POST":

        decision = request.form.get("decision", "").strip()
        comments = request.form.get("comments", "").strip()

        allowed_decisions = [
            "Accepted",
            "Requires Correction",
            "Rejected",
        ]

        if decision not in allowed_decisions:
            return "Invalid review decision.", 400

        try:

            review = submission.staff_review

            if review is None:

                review = StaffReview(
                    submission=submission,
                    decision=decision,
                    comments=comments
                )

                db.session.add(review)

            else:

                review.decision = decision
                review.comments = comments

            submission.workflow_status = decision

            db.session.commit()

            return redirect(url_for("dashboard"))

        except SQLAlchemyError:

            db.session.rollback()

            return "The review could not be saved. Please try again.", 500

        submission_time = None

    if submission.created_at is not None:

        stored_time = submission.created_at

        # SQLite stores CURRENT_TIMESTAMP in UTC.
        if stored_time.tzinfo is None:
            stored_time = stored_time.replace(tzinfo=timezone.utc)

        # Convert UTC to Namibia time (UTC+2).
        submission_time = stored_time.astimezone(
            timezone(timedelta(hours=2))
        )

    return render_template(
        "review_submission.html",
        submission=submission,
        submission_time=submission_time
    )



@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("home"))


# --------------------------------------------------
# 12. RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    with app.app_context():

        db.create_all()

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
