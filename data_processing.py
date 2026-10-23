from pathlib import Path
import csv

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "processed_data"
RESULTS_DIR = BASE_DIR / "eda_results" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Spark setup

spark = (
    SparkSession.builder
    .appName("CS236_Phase1_DataProcessing")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.default.parallelism", "4")
    .config("spark.sql.legacy.timeParserPolicy", "CORRECTED")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# Parse timestamps

def parse_timestamp(column_name):
    value = F.trim(F.col(column_name).cast("string"))

    return F.coalesce(
        F.to_timestamp(value, "yyyy-MM-dd'T'HH:mm:ss.SSS"),
        F.to_timestamp(value, "yyyy-MM-dd'T'HH:mm:ss"),
        F.to_timestamp(value, "yyyy-MM-dd HH:mm:ss"),
        F.to_timestamp(value, "MM/dd/yyyy hh:mm:ss a"),
        F.to_timestamp(value, "MM/dd/yyyy HH:mm:ss"),
        F.to_timestamp(value, "MM/dd/yyyy hh:mm a"),
        F.to_timestamp(value, "yyyy-MM-dd")
    )


# Load original datasets

def load_source(path, created_col, closed_col):
    df = spark.read.csv(
        str(path),
        header=True,
        inferSchema=False
    )

    for name in [created_col, closed_col]:
        df = df.withColumn(name + "_raw", F.col(name))
        df = df.withColumn(name, parse_timestamp(name))

    for name in ["latitude", "longitude"]:
        df = df.withColumn(
            name, F.col(name).cast("double")
        )

    for name in [
        "x_coordinate_state_plane",
        "y_coordinate_state_plane"
    ]:
        if name in df.columns:
            df = df.withColumn(
                name, F.col(name).cast("double")
            )

    return df


# Record data quality results

def count_and_record(df, condition, city, issue, results):
    value = df.filter(condition).count()

    results.append((city, issue, value))
    print(f"{issue}: {value}")

    return value


# Clean datasets

def clean_dataset(
    df, city, zip_col, created_col, closed_col, results
):
    print(f"\nCleaning {city} dataset")

    initial_count = df.count()
    print("Original records:", initial_count)

    # Clean string values

    for field in df.schema.fields:
        if isinstance(field.dataType, StringType):
            name = field.name
            stripped = F.trim(F.col(name))

            df = df.withColumn(
                name,
                F.when(stripped == "", None)
                .otherwise(stripped)
            )

    # Remove duplicates

    dedup_cols = [
        name for name in df.columns
        if not name.endswith("_raw")
    ]

    df = df.dropDuplicates(dedup_cols)

    cleaned_count = df.count()

    print("Records after deduplication:", cleaned_count)
    print("Duplicates removed:", initial_count - cleaned_count)

    # Clean postal codes

    df = df.withColumn(
        "postal_code_original",
        F.col(zip_col)
    )

    valid_format = F.col(zip_col).rlike(
        r"^[0-9]{5}(-[0-9]{4})?$"
    )

    df = df.withColumn(
        "postal_code_clean",
        F.when(
            valid_format,
            F.substring(F.col(zip_col), 1, 5)
        )
    )

    prefix = F.substring(
        F.col("postal_code_clean"), 1, 3
    ).cast("int")

    # Approximate city-specific ranges

    if city == "NYC":
        plausible_zip = (
            prefix.between(100, 104)
            | prefix.between(110, 114)
            | (prefix == 116)
        )

        lat_min, lat_max = 40.45, 40.95
        lon_min, lon_max = -74.30, -73.65

    else:
        plausible_zip = (
            prefix.between(900, 908)
            | prefix.between(910, 918)
        )

        lat_min, lat_max = 33.65, 34.40
        lon_min, lon_max = -118.75, -118.10

    df = df.withColumn(
        "suspicious_postal_code",
        F.coalesce(
            F.col("postal_code_clean").isNotNull()
            & (~plausible_zip),
            F.lit(False)
        )
    )

    # Check temporal ordering

    df = df.withColumn(
        "invalid_date_order",
        F.col(created_col).isNotNull()
        & F.col(closed_col).isNotNull()
        & (F.col(closed_col) < F.col(created_col))
    )

    # Check missing coordinates

    df = df.withColumn(
        "missing_coordinates",
        F.col("latitude").isNull()
        | F.col("longitude").isNull()
    )

    # Check invalid coordinates

    invalid_location = (
        (
            (F.col("latitude") == 0)
            & (F.col("longitude") == 0)
        )
        |
        (
            F.col("latitude").isNotNull()
            & F.col("longitude").isNotNull()
            & (
                (~F.col("latitude").between(lat_min, lat_max))
                | (~F.col("longitude").between(lon_min, lon_max))
            )
        )
    )

    df = df.withColumn(
        "invalid_coordinates",
        F.coalesce(invalid_location, F.lit(False))
    )

    # Validate cleaning

    print(f"\nData quality checks for {city}")

    checks = [
        (
            "missing_or_invalid_zip",
            F.col("postal_code_clean").isNull()
        ),
        (
            "suspicious_zip_prefix",
            F.col("suspicious_postal_code")
        ),
        (
            "invalid_date_order",
            F.col("invalid_date_order")
        ),
        (
            "missing_coordinates",
            F.col("missing_coordinates")
        ),
        (
            "coordinates_outside_rough_city_bounds",
            F.col("invalid_coordinates")
        )
    ]

    for issue, condition in checks:
        count_and_record(
            df, condition, city, issue, results
        )

    # Check timestamp parsing

    for name in [created_col, closed_col]:
        raw = name + "_raw"

        count_and_record(
            df,
            F.col(raw).isNotNull()
            & F.col(name).isNull(),
            city,
            "unparseable_" + name,
            results
        )

        count_and_record(
            df,
            F.col(name).isNull(),
            city,
            "null_" + name + "_after_cast",
            results
        )

    return df


# Adjust common schema

def adjust_schema(df, city):
    if city == "NYC":
        mapping = {
            "id": "unique_key",
            "created": "created_date",
            "closed": "closed_date",
            "type": "complaint_type",
            "agency": "agency",
            "zip": "postal_code_clean",
            "channel": "open_data_channel_type"
        }
    else:
        mapping = {
            "id": "srnumber",
            "created": "createddate",
            "closed": "closeddate",
            "type": "requesttype",
            "agency": "owner",
            "zip": "postal_code_clean",
            "channel": "requestsource"
        }

    return (
        df
        .withColumn(
            "original_request_id",
            F.col(mapping["id"]).cast("string")
        )
        .withColumn(
            "created_at",
            F.col(mapping["created"]).cast("timestamp")
        )
        .withColumn(
            "closed_at",
            F.col(mapping["closed"]).cast("timestamp")
        )
        .withColumn(
            "request_type",
            F.col(mapping["type"]).cast("string")
        )
        .withColumn(
            "status_original",
            F.col("status").cast("string")
        )
        .withColumn(
            "agency_common",
            F.col(mapping["agency"]).cast("string")
        )
        .withColumn(
            "postal_code",
            F.col(mapping["zip"]).cast("string")
        )
        .withColumn(
            "submission_channel_original",
            F.col(mapping["channel"]).cast("string")
        )
    )


# Validate common columns

COMMON_COLUMNS = [
    "original_request_id",
    "created_at",
    "closed_at",
    "request_type",
    "status_original",
    "agency_common",
    "postal_code",
    "postal_code_original",
    "submission_channel_original",
    "latitude",
    "longitude",
    "invalid_date_order",
    "missing_coordinates",
    "invalid_coordinates",
    "suspicious_postal_code"
]


def validate_common_schema(nyc_df, la_df):
    nyc_types = dict(nyc_df.dtypes)
    la_types = dict(la_df.dtypes)

    for name in COMMON_COLUMNS:
        if (
            name not in nyc_types
            or name not in la_types
            or nyc_types[name] != la_types[name]
        ):
            raise ValueError(
                f"Schema difference for {name}: "
                f"NYC={nyc_types.get(name)}, "
                f"LA={la_types.get(name)}"
            )

    print("SUCCESS: All common column types match.")


# Standardize categories

def standardize_categories(df, city):
    status = F.lower(F.trim(F.col("status_original")))
    channel = F.lower(
        F.trim(F.col("submission_channel_original"))
    )

    df = df.withColumn(
        "status_standardized",
        F.when(status == "closed", "Closed")
        .when(status == "open", "Open")
        .when(status == "pending", "Pending")
        .when(
            status.isin("in progress", "assigned", "started"),
            "In Progress"
        )
        .when(status == "cancelled", "Cancelled")
        .when(status == "referred out", "Referred Out")
        .when(status == "forward", "Forward")
        .when(
            status.isNull()
            | status.isin("unknown", "unspecified"),
            "Unknown"
        )
        .otherwise(F.col("status_original"))
    )

    df = df.withColumn(
        "submission_channel_standardized",
        F.when(
            channel.isNull()
            | channel.isin("unknown", "unspecified"),
            "Unknown"
        )
        .when(
            channel.isin("phone", "call", "voicemail"),
            "Phone"
        )
        .when(
            channel.isin("mobile", "mobile app"),
            "Mobile"
        )
        .when(
            channel.isin("online", "self service", "web form"),
            "Online / Self-service"
        )
        .when(
            channel.isin("email", "fax", "letter"),
            "Email / Mail / Fax"
        )
        .when(
            channel.isin("walk-in", "walk in"),
            "In-person"
        )
        .when(channel == "social", "Social Media")
        .when(
            channel == "driver self report",
            "Driver Self Report"
        )
        .when(
            channel.isin(
                "council's office",
                "council’s office",
                "city attorney",
                "queue initiated customer call",
                "radio"
            ),
            "City Staff / Internal"
        )
        .otherwise("Other")
    )

    print(f"\n{city} status distribution")

    (
        df.groupBy("status_standardized")
        .count()
        .orderBy(F.desc("count"), "status_standardized")
        .show(30, False)
    )

    print(f"\n{city} channel distribution")

    (
        df.groupBy("submission_channel_standardized")
        .count()
        .orderBy(
            F.desc("count"),
            "submission_channel_standardized"
        )
        .show(30, False)
    )

    return df


# Check temporal inconsistencies

def temporal_checks(df, city, results):
    print(f"\nTemporal consistency for {city}")

    status = F.col("status_standardized")

    checks = [
        (
            "closed_missing_closure_date",
            (status == "Closed") & F.col("closed_at").isNull()
        ),
        (
            "open_with_closure_date",
            (status == "Open") & F.col("closed_at").isNotNull()
        ),
        (
            "future_created_at",
            F.col("created_at") > F.current_timestamp()
        ),
        (
            "future_closed_at",
            F.col("closed_at") > F.current_timestamp()
        )
    ]

    for issue, condition in checks:
        count_and_record(
            df, condition, city, issue, results
        )

    return df


# Create required attributes

def create_attributes(df, city):
    df = (
        df
        .withColumn("source_city", F.lit(city))
        .withColumn(
            "global_request_id",
            F.concat(
                F.lit(city + "_"),
                F.col("original_request_id")
            )
        )
        .withColumn(
            "resolution_hours",
            F.when(
                F.col("created_at").isNotNull()
                & F.col("closed_at").isNotNull()
                & (
                    F.col("closed_at") >= F.col("created_at")
                ),
                (
                    F.unix_timestamp("closed_at")
                    - F.unix_timestamp("created_at")
                ) / 3600.0
            ).otherwise(F.lit(None).cast("double"))
        )
    )

    # Flag completed requests with valid timestamps

    df = df.withColumn(
        "is_valid_completed",
        (
            (F.col("status_standardized") == "Closed")
            & F.col("created_at").isNotNull()
            & F.col("closed_at").isNotNull()
            & (F.col("closed_at") >= F.col("created_at"))
        )
    )

    # Validate derived attributes

    total = df.count()

    unique = df.select(
        "global_request_id"
    ).distinct().count()

    missing = df.filter(
        F.col("global_request_id").isNull()
        | (F.length("original_request_id") == 0)
    ).count()

    negative = df.filter(
        F.col("resolution_hours") < 0
    ).count()

    invalid_completed = df.filter(
        F.col("is_valid_completed")
        & (
            F.col("resolution_hours").isNull()
            | (F.col("status_standardized") != "Closed")
        )
    ).count()

    completed_count = df.filter(
        F.col("is_valid_completed")
    ).count()

    print(f"\n{city} attribute validation")
    print("Total records:", total)
    print("Distinct global IDs:", unique)
    print("Missing global IDs:", missing)
    print("Negative resolution hours:", negative)
    print("Valid completed requests:", completed_count)
    print("Invalid completed flags:", invalid_completed)

    if unique != total or missing or negative or invalid_completed:
        raise ValueError(
            f"Failed {city} attribute validation"
        )

    print(
        "NULL resolution hours:",
        df.filter(
            F.col("resolution_hours").isNull()
        ).count()
    )

    print(f"SUCCESS: {city} attributes validated.")

    return df


# Save and validate Parquet datasets

def persist_and_validate(df, city, name):
    path = OUTPUT_DIR / name

    print(f"\nSaving {city} dataset")

    expected_count = df.count()
    expected_schema = dict(df.dtypes)

    df.write.mode("overwrite").parquet(str(path))

    saved = spark.read.parquet(str(path))

    saved_count = saved.count()
    saved_schema = dict(saved.dtypes)

    print("Expected records:", expected_count)
    print("Reloaded records:", saved_count)

    if saved_count != expected_count:
        raise ValueError(
            f"{city} saved record count mismatch"
        )

    if saved_schema != expected_schema:
        raise ValueError(
            f"{city} saved schema mismatch"
        )

    wrong_city = saved.filter(
        F.col("source_city") != city
    ).count()

    unique_count = saved.select(
        "global_request_id"
    ).distinct().count()

    null_flags = saved.filter(
        F.col("invalid_coordinates").isNull()
        | F.col("is_valid_completed").isNull()
    ).count()

    if wrong_city or unique_count != expected_count or null_flags:
        raise ValueError(
            f"{city} saved validation failed"
        )

    print(f"SUCCESS: {city} saved and verified.")


# Save data quality summary

def save_quality_results(results):
    path = RESULTS_DIR / "data_quality_summary.csv"

    with open(
        path, "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["city", "check", "count"])
        writer.writerows(results)

    print("Saved quality summary:", path)


# Run data processing

try:
    quality_results = []

    nyc = load_source(
        DATA_DIR / "nyc_311_800k.csv",
        "created_date",
        "closed_date"
    )

    la = load_source(
        DATA_DIR / "la_311_800k.csv",
        "createddate",
        "closeddate"
    )

    print(
        "Postal-code input types:",
        dict(nyc.dtypes)["incident_zip"],
        dict(la.dtypes)["zipcode"]
    )

    nyc = clean_dataset(
        nyc, "NYC", "incident_zip",
        "created_date", "closed_date",
        quality_results
    )

    la = clean_dataset(
        la, "LA", "zipcode",
        "createddate", "closeddate",
        quality_results
    )

    nyc = adjust_schema(nyc, "NYC")
    la = adjust_schema(la, "LA")

    validate_common_schema(nyc, la)

    nyc = standardize_categories(nyc, "NYC")
    la = standardize_categories(la, "LA")

    nyc = temporal_checks(
        nyc, "NYC", quality_results
    )

    la = temporal_checks(
        la, "LA", quality_results
    )

    nyc = create_attributes(nyc, "NYC")
    la = create_attributes(la, "LA")

    persist_and_validate(
        nyc, "NYC", "nyc_cleaned"
    )

    persist_and_validate(
        la, "LA", "la_cleaned"
    )

    save_quality_results(quality_results)

    print("\nSUCCESS: Data processing completed.")

finally:
    spark.stop()
