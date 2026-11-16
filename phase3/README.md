# CS236 Phase 3: WebUI for Searching, Filtering, and Data Entry

A small Flask web app that connects directly to the PostgreSQL database built in
Phase 2. Every search and every insert is sent to PostgreSQL as a SQL query.

- **Browse** (`/`): pick a dataset (unified, NYC cleaned, LA cleaned), filter it,
  and page through the records.
- **New request** (`/new`): form that inserts a new service request into the
  unified table (`merged_311`).

## Prerequisites

1. The Phase 2 database is running and loaded (tables `nyc_cleaned`,
   `la_cleaned`, `merged_311`). If it is not, follow the Phase 2 steps:
   start the container, create the tables, and run `phase2/load_postgres.py`.
2. Docker Desktop is open and the container is up:
   ```bash
   docker ps --filter name=cs236-postgres
   ```
   If it shows `Exited`, start it with `docker start cs236-postgres`.
3. Python 3.11 with the project virtual environment (`.venv`).

## Setup (once)

From the project folder:

```bash
source .venv/bin/activate
```
```bash
pip install -r phase3/requirements.txt
```

This installs Flask and psycopg2 (the PostgreSQL driver for Python).

## Run

```bash
export CS236_DB_PASSWORD=<your database password>
```
```bash
python phase3/app.py
```

Open **http://127.0.0.1:5001** in a browser. Stop the app with `Ctrl+C`.

The password is read from the `CS236_DB_PASSWORD` environment variable (the same
one `phase2/load_postgres.py` uses), so it is never stored in the code. The app
connects to `127.0.0.1:5432`, database `cs236_project`, user `cs236`; these
constants are at the top of `app.py`.

## Using the WebUI

**Dataset selection.** The buttons at the top switch between *Unified*
(`merged_311`), *NYC cleaned* (`nyc_cleaned`), and *LA cleaned* (`la_cleaned`).

**Filtering.** Fill in any combination of these and press *Search*. Every filter
you fill in is added to the same query with `AND` (combined filters):

| Filter | Matches |
|---|---|
| City | exact city (NYC / LA) |
| Created from / to | creation date range, both days included |
| Status | standardized status |
| Agency | responsible agency |
| Submission channel | standardized channel |
| Request type | contains the text, ignoring case |
| Postal code | exact 5-digit code |
| Request ID | exact request ID (for example `LA_1-4604958861`) |

**Result management.** Only 50 records are fetched per page (`LIMIT`/`OFFSET`),
sorted newest first. Use *Previous* / *Next* to move between pages; filters stay
applied. The full table is never loaded.

**SQL box.** Under each result table the page shows the exact SQL that was sent
to PostgreSQL for that page.

**New request.** Fill in the form at `/new` and press *Save request*:

- Required: city, request type, created time, status, agency, submission channel.
- Optional: closed time, postal code, latitude and longitude (give both or neither).
- The request ID is generated automatically (for example `NYC_WEB-3fa9c01b27de`),
  and the primary key guarantees it is unique.
- Rejected: missing required fields, dates in the future, closed before created,
  a postal code that is not 5 digits, impossible coordinates, or an agency from the
  other city.
- Stored but flagged (same rules as Phase 1): a postal code outside the city's
  usual range sets `suspicious_postal_code`; coordinates outside the city's
  bounding box set `invalid_coordinates`.
- `resolution_hours` and `is_valid_completed` are calculated the same way as in
  Phase 1.

After saving, the app shows the stored record read back from the database.

## How the app talks to PostgreSQL

- All values the user types are sent as query **parameters** (`%s`), never pasted
  into the SQL text, so input cannot change the query.
- Table names come only from a fixed list of three tables in `app.py`.
- The browse page opens a **read-only** connection, so it cannot change data.
  Only the *New request* form writes, and only with one `INSERT` into `merged_311`.
- Dropdown choices are read once from the unified table and cached; the server
  loads them when it starts.

## Removing test records

Records added through the form live in `merged_311`, so the table will hold more
than 1,600,000 rows afterwards. To remove only the records created by the form:

```bash
docker exec -it cs236-postgres psql -U cs236 -d cs236_project -c "DELETE FROM merged_311 WHERE original_request_id LIKE 'WEB-%';"
```

## Troubleshooting

| Message or problem | Fix |
|---|---|
| "The database password is not set" | Run the `export CS236_DB_PASSWORD=...` command, then restart the app. |
| "Could not connect to PostgreSQL" | Open Docker Desktop and run `docker start cs236-postgres`; check the password. |
| Port 5432 already in use when starting Docker | Stop any other local PostgreSQL (for example `brew services stop postgresql@14`). |
| Port 5001 already in use | Change `WEB_PORT` at the top of `app.py`. (Port 5000 is avoided because macOS uses it for AirPlay.) |

## Files

| File | Purpose |
|---|---|
| `app.py` | Flask app: routes, SQL queries, validation, insert |
| `templates/base.html` | Page layout and styling |
| `templates/index.html` | Browse page: dataset buttons, filters, table, pagination |
| `templates/new_request.html` | New-request form |
| `requirements.txt` | Python packages (Flask, psycopg2-binary) |
