"""
CS236 Phase 3: lightweight WebUI for searching, filtering, and data entry.

The app connects directly to the PostgreSQL database populated in Phase 2 and
sends every search and every insert to PostgreSQL as a parameterized SQL
query. The full tables are never loaded into Python; only the one page of
rows being displayed is fetched.

Pages
  /      Browse one of the three tables (unified, NYC cleaned, LA cleaned),
         filter the records, and page through the results.
  /new   Form that inserts a new service request into the unified table.

Run (full setup in README.md):
  export CS236_DB_PASSWORD=<password>
  python phase3/app.py        then open http://127.0.0.1:5001
"""

import os
import re
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta

import psycopg2
from flask import Flask, redirect, render_template, request, url_for
from psycopg2 import errors, sql

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Same connection settings as phase2/load_postgres.py. The password comes from
# the environment so it never appears in the source code.
DB_HOST = "127.0.0.1"
DB_PORT = 5432
DB_NAME = "cs236_project"
DB_USER = "cs236"
DB_PASSWORD = os.environ.get("CS236_DB_PASSWORD")

# macOS reserves port 5000 for AirPlay Receiver, so the app uses 5001.
WEB_PORT = 5001

# Result management: show this many records per page.
PAGE_SIZE = 50
MAX_PAGE = 1_000_000

# Dataset selection: URL key -> (label shown to the user, PostgreSQL table).
# Table names are only ever taken from this dictionary, never from user input.
DATASETS = {
    "unified": ("Unified", "merged_311"),
    "nyc": ("NYC cleaned", "nyc_cleaned"),
    "la": ("LA cleaned", "la_cleaned"),
}
UNIFIED_TABLE = "merged_311"

# Columns shown in the results table as (database column, heading). All three
# tables contain these columns with the same meaning, so one list works for
# every dataset.
DISPLAY_COLUMNS = [
    ("global_request_id", "Request ID"),
    ("source_city", "City"),
    ("created_at", "Created"),
    ("closed_at", "Closed"),
    ("request_type", "Request type"),
    ("status_standardized", "Status"),
    ("agency_common", "Agency"),
    ("postal_code", "Postal code"),
    ("submission_channel_standardized", "Channel"),
    ("resolution_hours", "Resolution (hours)"),
]

# Filters accepted on the browse page (names of the form fields).
FILTER_NAMES = [
    "request_id", "source_city", "date_from", "date_to", "status",
    "agency", "request_type", "channel", "postal_code",
]

# City rules copied from Phase 1 (data_processing.py) so that records entered
# through the form are flagged exactly like the cleaned data: approximate city
# bounding box, and the ZIP-code prefixes (first three digits) that are
# plausible for the city.
CITY_RULES = {
    "NYC": {
        "lat": (40.45, 40.95),
        "lon": (-74.30, -73.65),
        "zip_prefixes": [(100, 104), (110, 114), (116, 116)],
    },
    "LA": {
        "lat": (33.65, 34.40),
        "lon": (-118.75, -118.10),
        "zip_prefixes": [(900, 908), (910, 918)],
    },
}

# Fields on the "new request" form.
NEW_FIELDS = [
    "source_city", "request_type", "created_at", "closed_at", "status",
    "agency", "channel", "postal_code", "latitude", "longitude",
]

# Defaults pre-selected on the empty form.
FORM_DEFAULTS = {
    "source_city": "NYC",
    "status": "Open",
    "channel": "Online / Self-service",
}

# The 21 columns of the unified table (merged_311), in table order.
INSERT_COLUMNS = [
    "source_city", "global_request_id", "original_request_id", "created_at",
    "closed_at", "request_type", "status_original", "status_standardized",
    "agency_common", "postal_code", "postal_code_original",
    "suspicious_postal_code", "submission_channel_original",
    "submission_channel_standardized", "latitude", "longitude",
    "invalid_date_order", "missing_coordinates", "invalid_coordinates",
    "resolution_hours", "is_valid_completed",
]

app = Flask(__name__)

# Which city's rows each cleaned table holds (used for its dropdown choices).
DATASET_CITY = {"nyc": "NYC", "la": "LA"}

# Dropdown choices are looked up once and reused, so the page does not rescan
# 1.6 million rows on every request.
_options_cache = {}


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

class DatabaseUnavailable(Exception):
    """The database password is missing or PostgreSQL cannot be reached."""


@contextmanager
def db_connection(readonly=False):
    """Open a PostgreSQL connection for one request and always close it.

    readonly=True makes PostgreSQL itself reject any write on the connection,
    which guarantees that the browse page can never modify data.
    """
    if not DB_PASSWORD:
        raise DatabaseUnavailable(
            "The database password is not set. Stop the app, run "
            "'export CS236_DB_PASSWORD=<your password>', and start it again."
        )
    try:
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
            user=DB_USER, password=DB_PASSWORD, connect_timeout=5,
        )
    except psycopg2.OperationalError as exc:
        raise DatabaseUnavailable(
            "Could not connect to PostgreSQL at "
            f"{DB_HOST}:{DB_PORT}. Check that the Docker container is running "
            f"and that the password is correct. ({str(exc).strip()})"
        ) from exc
    try:
        conn.set_session(readonly=readonly)
        yield conn
    finally:
        conn.close()


def get_options(conn):
    """Distinct status, agency, channel and request-type values (cached).

    One scan of the unified table (GROUP BY city) supplies the dropdown choices
    for all three datasets. This is valid because the unified table holds every
    record of both cleaned tables, and it is much cheaper than scanning the
    wide cleaned tables once per dropdown.

    Returns {"by_dataset": {"unified"|"nyc"|"la": {"status": [...],
    "agency": [...], "channel": [...], "request_type": [...]}},
    "agencies_by_city": {"NYC": [...], "LA": [...]}}.
    """
    if "options" not in _options_cache:
        fields = ["status", "agency", "channel", "request_type"]
        per_city = {}
        with conn.cursor() as cur:
            cur.execute(
                "SELECT source_city,\n"
                "  array_agg(DISTINCT status_standardized) FILTER (WHERE status_standardized IS NOT NULL),\n"
                "  array_agg(DISTINCT agency_common) FILTER (WHERE agency_common IS NOT NULL),\n"
                "  array_agg(DISTINCT submission_channel_standardized) FILTER (WHERE submission_channel_standardized IS NOT NULL),\n"
                "  array_agg(DISTINCT request_type) FILTER (WHERE request_type IS NOT NULL)\n"
                "FROM merged_311 GROUP BY source_city"
            )
            for city, *lists in cur.fetchall():
                per_city[city] = {
                    field: sorted(values or [], key=str.casefold)
                    for field, values in zip(fields, lists)
                }

        empty = {field: [] for field in fields}
        by_dataset = {
            "unified": {
                field: sorted({v for opts in per_city.values() for v in opts[field]},
                              key=str.casefold)
                for field in fields
            },
        }
        for dataset, city in DATASET_CITY.items():
            by_dataset[dataset] = per_city.get(city, empty)

        _options_cache["options"] = {
            "by_dataset": by_dataset,
            "agencies_by_city": {
                city: per_city.get(city, empty)["agency"] for city in CITY_RULES
            },
        }
    return _options_cache["options"]


@app.errorhandler(DatabaseUnavailable)
def handle_database_unavailable(exc):
    return render_template("base.html", error=str(exc)), 503


@app.errorhandler(psycopg2.Error)
def handle_database_error(exc):
    return render_template("base.html", error=f"Database error: {exc}"), 500


# ---------------------------------------------------------------------------
# Browse page: dataset selection, retrieval, filtering, pagination
# ---------------------------------------------------------------------------

def parse_page(text):
    """Page number from the URL, forced into a sensible range."""
    try:
        page = int(text)
    except (TypeError, ValueError):
        return 1
    return max(1, min(page, MAX_PAGE))


def parse_date(text, label, messages):
    """Parse a yyyy-mm-dd filter value; report and ignore a bad one."""
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        messages.append(f"'{label}' must be a date like 2024-03-15, so that filter was ignored.")
        return None


def escape_like(text):
    """Make %, _ and \\ in what the user typed match literally in ILIKE."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def build_where(filters, messages):
    """Turn the filled-in filters into a WHERE clause plus its parameters.

    Every filter the user filled in adds one condition, and all conditions are
    combined with AND (combined filters). User-entered values are never pasted
    into the SQL text; they are sent separately as %s parameters.
    """
    clauses, params = [], []

    def add(condition, value):
        clauses.append(sql.SQL(condition))
        params.append(value)

    if filters["request_id"]:
        add("global_request_id = %s", filters["request_id"])
    if filters["source_city"]:
        add("source_city = %s", filters["source_city"])

    date_from = parse_date(filters["date_from"], "Created from", messages)
    date_to = parse_date(filters["date_to"], "Created to", messages)
    if date_from:
        add("created_at >= %s", date_from)
    if date_to:
        # "to" is inclusive of that whole day.
        add("created_at < %s", date_to + timedelta(days=1))
    if date_from and date_to and date_from > date_to:
        messages.append("'Created from' is after 'Created to', so no records can match.")

    if filters["status"]:
        add("status_standardized = %s", filters["status"])
    if filters["agency"]:
        add("agency_common = %s", filters["agency"])
    if filters["request_type"]:
        add("request_type ILIKE %s", "%" + escape_like(filters["request_type"]) + "%")
    if filters["channel"]:
        add("submission_channel_standardized = %s", filters["channel"])
    if filters["postal_code"]:
        add("postal_code = %s", filters["postal_code"])

    if not clauses:
        return sql.SQL(""), params
    where = sql.SQL("\nWHERE ") + sql.SQL("\n  AND ").join(clauses)
    return where, params


def format_cell(value):
    """Readable text for one table cell."""
    if value is None:
        return "–"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


@app.route("/")
def index():
    # Dataset selection: anything unknown falls back to the unified table.
    dataset = request.args.get("dataset", "unified")
    if dataset not in DATASETS:
        dataset = "unified"
    dataset_label, table = DATASETS[dataset]

    filters = {name: request.args.get(name, "").strip() for name in FILTER_NAMES}
    page = parse_page(request.args.get("page"))
    messages = []
    where, params = build_where(filters, messages)

    # Result management: fetch one extra row so we know whether a next page
    # exists, without counting or loading the whole table.
    query = sql.SQL(
        "SELECT {cols}\nFROM {table}{where}\n"
        "ORDER BY created_at DESC, global_request_id\nLIMIT %s OFFSET %s"
    ).format(
        cols=sql.SQL(", ").join(sql.Identifier(c) for c, _ in DISPLAY_COLUMNS),
        table=sql.Identifier(table),
        where=where,
    )
    query_params = params + [PAGE_SIZE + 1, (page - 1) * PAGE_SIZE]

    with db_connection(readonly=True) as conn:
        options = get_options(conn)["by_dataset"][dataset]
        with conn.cursor() as cur:
            shown_sql = cur.mogrify(query, query_params).decode()
            cur.execute(query, query_params)
            fetched = cur.fetchall()

    has_next = len(fetched) > PAGE_SIZE
    rows = [[format_cell(v) for v in row] for row in fetched[:PAGE_SIZE]]

    # Previous/Next links keep the dataset and every filter value.
    nav = {"dataset": dataset, **{k: v for k, v in filters.items() if v}}
    prev_url = url_for("index", **nav, page=page - 1) if page > 1 else None
    next_url = url_for("index", **nav, page=page + 1) if has_next else None
    first_row = (page - 1) * PAGE_SIZE + 1

    return render_template(
        "index.html",
        datasets=[(key, label) for key, (label, _) in DATASETS.items()],
        dataset=dataset,
        dataset_label=dataset_label,
        table=table,
        filters=filters,
        options=options,
        cities=list(CITY_RULES),
        columns=[heading for _, heading in DISPLAY_COLUMNS],
        rows=rows,
        page=page,
        page_size=PAGE_SIZE,
        first_row=first_row,
        last_row=first_row + len(rows) - 1,
        prev_url=prev_url,
        next_url=next_url,
        shown_sql=shown_sql,
        messages=messages,
        inserted=request.args.get("inserted") == "1",
    )


# ---------------------------------------------------------------------------
# New-request page: validation and INSERT into the unified table
# ---------------------------------------------------------------------------

def form_choices(conn):
    """Dropdown and suggestion lists for the form, read from the database."""
    options = get_options(conn)
    unified = options["by_dataset"]["unified"]
    return {
        "statuses": unified["status"],
        "channels": unified["channel"],
        "request_types": unified["request_type"],
        "agencies": options["agencies_by_city"],
    }


def parse_timestamp(text, label, problems, required=False):
    """Parse a date-time from the form; record a problem if it is unusable."""
    if not text:
        if required:
            problems.append(f"{label} time is required.")
        return None
    try:
        value = datetime.fromisoformat(text).replace(tzinfo=None)
    except ValueError:
        problems.append(f"{label} time is not a valid date and time.")
        return None
    if value > datetime.now():
        problems.append(f"{label} time cannot be in the future.")
    return value


def validate_new_request(form, choices):
    """Check the submitted form and build the row to insert.

    Returns (values, record, problems):
      values   the raw text the user typed, to refill the form after an error
      record   dict of typed column values for merged_311 (None if invalid)
      problems list of messages to show the user (empty if valid)

    Rules: impossible input is rejected (missing required field, bad date,
    closed before created, future date, ZIP that is not 5 digits, coordinates
    that are not valid latitude/longitude, agency from the wrong city).
    Plausible-but-suspicious input is stored and flagged, as in Phase 1: a ZIP
    outside the city's usual range sets suspicious_postal_code, and coordinates
    outside the city's bounding box set invalid_coordinates.
    """
    values = {name: form.get(name, "").strip() for name in NEW_FIELDS}
    problems = []

    city = values["source_city"]
    if city not in CITY_RULES:
        problems.append("Choose a city (NYC or LA).")

    request_type = values["request_type"]
    if not request_type:
        problems.append("Request type is required.")
    elif len(request_type) > 200:
        problems.append("Request type must be 200 characters or fewer.")

    created_at = parse_timestamp(values["created_at"], "Created", problems, required=True)
    closed_at = parse_timestamp(values["closed_at"], "Closed", problems)
    if created_at and closed_at and closed_at < created_at:
        problems.append("Closed time cannot be earlier than the created time.")

    status = values["status"]
    if status not in choices["statuses"]:
        problems.append("Choose a status from the list.")

    agency = values["agency"]
    if city in CITY_RULES and agency not in choices["agencies"].get(city, []):
        problems.append(f"Choose an agency that belongs to {city}.")

    channel = values["channel"]
    if channel not in choices["channels"]:
        problems.append("Choose a submission channel from the list.")

    postal_code = values["postal_code"]
    if postal_code and not re.fullmatch(r"[0-9]{5}", postal_code):
        problems.append("Postal code must be exactly 5 digits (for example 90026).")

    latitude = longitude = None
    if values["latitude"] or values["longitude"]:
        try:
            latitude = float(values["latitude"])
            longitude = float(values["longitude"])
        except ValueError:
            problems.append("Latitude and longitude must both be numbers, or both left blank.")
        else:
            if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                problems.append(
                    "Latitude must be between -90 and 90 and longitude between -180 and 180."
                )

    if problems:
        return values, None, problems

    # Quality flags, computed the same way as in Phase 1.
    rules = CITY_RULES[city]
    suspicious_postal_code = False
    if postal_code:
        prefix = int(postal_code[:3])
        suspicious_postal_code = not any(lo <= prefix <= hi for lo, hi in rules["zip_prefixes"])
    invalid_coordinates = False
    if latitude is not None:
        invalid_coordinates = not (
            rules["lat"][0] <= latitude <= rules["lat"][1]
            and rules["lon"][0] <= longitude <= rules["lon"][1]
        )

    resolution_hours = None
    if closed_at is not None:
        resolution_hours = (closed_at - created_at).total_seconds() / 3600.0

    # Unique identifier: a random id with the same CITY_ prefix convention as
    # the Phase 1 global_request_id. The primary key enforces uniqueness.
    original_request_id = "WEB-" + uuid.uuid4().hex[:12]

    record = {
        "source_city": city,
        "global_request_id": f"{city}_{original_request_id}",
        "original_request_id": original_request_id,
        "created_at": created_at,
        "closed_at": closed_at,
        "request_type": request_type,
        "status_original": status,
        "status_standardized": status,
        "agency_common": agency,
        "postal_code": postal_code or None,
        "postal_code_original": postal_code or None,
        "suspicious_postal_code": suspicious_postal_code,
        "submission_channel_original": channel,
        "submission_channel_standardized": channel,
        "latitude": latitude,
        "longitude": longitude,
        "invalid_date_order": False,
        "missing_coordinates": latitude is None,
        "invalid_coordinates": invalid_coordinates,
        "resolution_hours": resolution_hours,
        "is_valid_completed": status == "Closed" and closed_at is not None,
    }
    return values, record, problems


def insert_request(conn, record):
    """INSERT one record into the unified table (the caller commits)."""
    query = sql.SQL("INSERT INTO {table} ({cols}) VALUES ({marks})").format(
        table=sql.Identifier(UNIFIED_TABLE),
        cols=sql.SQL(", ").join(sql.Identifier(c) for c in INSERT_COLUMNS),
        marks=sql.SQL(", ").join([sql.Placeholder()] * len(INSERT_COLUMNS)),
    )
    with conn.cursor() as cur:
        cur.execute(query, [record[c] for c in INSERT_COLUMNS])


@app.route("/new", methods=["GET", "POST"])
def new_request():
    with db_connection() as conn:
        choices = form_choices(conn)
        problems = []

        if request.method == "POST":
            values, record, problems = validate_new_request(request.form, choices)
            if not problems:
                try:
                    insert_request(conn, record)
                    conn.commit()
                except errors.UniqueViolation:
                    conn.rollback()
                    problems.append("That request ID is already taken. Please submit again.")
                else:
                    # A new request type may now exist; refresh suggestions.
                    _options_cache.clear()
                    return redirect(url_for(
                        "index", dataset="unified",
                        request_id=record["global_request_id"], inserted=1,
                    ))
        else:
            values = dict(FORM_DEFAULTS)
            values["created_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M")

    status_code = 400 if problems else 200
    return render_template(
        "new_request.html", values=values, choices=choices,
        cities=list(CITY_RULES), problems=problems,
    ), status_code


if __name__ == "__main__":
    # Load the dropdown choices before the first page request so the first
    # page load is quick. If the database is not reachable yet, start anyway;
    # the pages will explain the problem.
    try:
        with db_connection(readonly=True) as conn:
            get_options(conn)
    except (DatabaseUnavailable, psycopg2.Error) as exc:
        print(f"Warning: could not preload dropdown choices: {exc}")

    app.run(host="127.0.0.1", port=WEB_PORT, debug=False)
