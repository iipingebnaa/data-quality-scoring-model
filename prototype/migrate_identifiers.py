
from pathlib import Path
import sqlite3


# --------------------------------------------------
# 1. DATABASE LOCATIONS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATABASE_FILE = PROJECT_DIR / "data" / "onboarding.db"

BACKUP_FILE = (
    PROJECT_DIR
    / "data"
    / "onboarding_backup_before_identifier_migration.db"
)


# --------------------------------------------------
# 2. CHECK DATABASE AND BACKUP
# --------------------------------------------------

if not DATABASE_FILE.exists():
    raise FileNotFoundError("The original database was not found.")

if not BACKUP_FILE.exists():
    raise FileNotFoundError("The database backup was not found.")


# --------------------------------------------------
# 3. MIGRATE EXISTING IDENTIFIERS
# --------------------------------------------------

connection = sqlite3.connect(DATABASE_FILE)

try:

    connection.execute("BEGIN IMMEDIATE")

    customer_count_before = connection.execute(
        "SELECT COUNT(*) FROM customer"
    ).fetchone()[0]

    submission_count_before = connection.execute(
        "SELECT COUNT(*) FROM submission"
    ).fetchone()[0]

    review_count_before = connection.execute(
        "SELECT COUNT(*) FROM staff_review"
    ).fetchone()[0]

    # Retrieve customers in their original registration order.

    customers = connection.execute(
        """
        SELECT id
        FROM customer
        ORDER BY id ASC
        """
    ).fetchall()

    # Assign sequential customer numbers.

    for position, (customer_id,) in enumerate(customers, start=1):

        new_number = f"CUS-{1000 + position}"

        connection.execute(
            """
            UPDATE customer
            SET customer_number = ?
            WHERE id = ?
            """,
            (new_number, customer_id)
        )

    # Retrieve submissions in their original creation order.

    submissions = connection.execute(
        """
        SELECT id
        FROM submission
        ORDER BY id ASC
        """
    ).fetchall()

    # Assign sequential submission references.

    for position, (submission_id,) in enumerate(
        submissions, start=1
    ):

        new_reference = f"SUB-{10000 + position}"

        connection.execute(
            """
            UPDATE submission
            SET submission_reference = ?
            WHERE id = ?
            """,
            (new_reference, submission_id)
        )

    # Verify that no records were lost.

    customer_count_after = connection.execute(
        "SELECT COUNT(*) FROM customer"
    ).fetchone()[0]

    submission_count_after = connection.execute(
        "SELECT COUNT(*) FROM submission"
    ).fetchone()[0]

    review_count_after = connection.execute(
        "SELECT COUNT(*) FROM staff_review"
    ).fetchone()[0]

    if (
        customer_count_before != customer_count_after
        or submission_count_before != submission_count_after
        or review_count_before != review_count_after
    ):
        raise RuntimeError(
            "Record counts changed. Migration cancelled."
        )

    connection.commit()

    print("\nIDENTIFIER MIGRATION SUCCESSFUL")

    print(f"Customers preserved: {customer_count_after}")
    print(f"Submissions preserved: {submission_count_after}")
    print(f"Staff reviews preserved: {review_count_after}")

    print("\nUPDATED CUSTOMER NUMBERS:")

    for row in connection.execute(
        """
        SELECT id, customer_number
        FROM customer
        ORDER BY id
        """
    ):
        print(row)

    print("\nUPDATED SUBMISSION REFERENCES:")

    for row in connection.execute(
        """
        SELECT id, submission_reference
        FROM submission
        ORDER BY id
        """
    ):
        print(row)

except Exception:

    connection.rollback()

    print("Migration failed. All changes were rolled back.")

    raise

finally:

    connection.close()
