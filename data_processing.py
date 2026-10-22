from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

OUTPUT_DIR = BASE_DIR / "processed_data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Spark setup

spark = (
    SparkSession.builder
    .appName("CS236_Phase1_DataProcessing")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.default.parallelism", "4")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Display the maximum available JVM heap.
jvm = spark.sparkContext._jvm

heap_gb = (
    jvm.java.lang.Runtime.getRuntime().maxMemory()
    / (1024 ** 3)
)

print(f"\nSpark JVM maximum heap: {heap_gb:.2f} GB")

print(f"Project directory: {BASE_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

# Data cleaning

def clean_dataset(
    df,
    dataset_name,
    zip_col,
    created_col,
    closed_col
):
    """
    Apply initial cleaning based on EDA findings.

    Operations:
    1. Remove exact duplicate records.
    2. Trim text values and convert blanks to NULL.
    3. Preserve and validate postal codes.
    4. Flag invalid timestamp ordering.
    5. Flag missing coordinates.

    Preserve rows with missing closure dates.
    """

    print(
        f"\n========== CLEANING {dataset_name} =========="
    )

    original_count = df.count()

    print("Original records:", original_count)

# Remove Duplicates

    df = df.dropDuplicates()

    deduplicated_count = df.count()

    print(
        "Records after deduplication:",
        deduplicated_count
    )

    print(
        "Duplicates removed:",
        original_count - deduplicated_count
    )

# Clean String Values

    for field in df.schema.fields:

        if isinstance(field.dataType, StringType):

            column_name = field.name

            trimmed_value = F.trim(
                F.col(column_name)
            )

            df = df.withColumn(
                column_name,
                F.when(
                    trimmed_value == "",
                    F.lit(None)
                ).otherwise(trimmed_value)
            )

# Clean Postal Codes
# Preserve the source value after whitespace cleanup
    df = df.withColumn(
        "postal_code_original",
        F.col(zip_col)
    )

    valid_zip = F.col(zip_col).rlike(
        r"^[0-9]{5}(-[0-9]{4})?$"
    )

    df = df.withColumn(
        "postal_code_clean",
        F.when(
            valid_zip,
            F.col(zip_col)
        ).otherwise(
            F.lit(None).cast("string")
        )
    )

# Check if the closing date is before the creation date

    df = df.withColumn(
        "invalid_date_order",
        F.when(
            F.col(created_col).isNotNull()
            & F.col(closed_col).isNotNull()
            & (
                F.col(closed_col)
                < F.col(created_col)
            ),
            F.lit(True)
        ).otherwise(F.lit(False))
    )

# Check for missing coordinates

    df = df.withColumn(
        "missing_coordinates",
        F.col("latitude").isNull()
        | F.col("longitude").isNull()
    )

# Validate cleaned data

    print("\nCLEANING VALIDATION:")

    print("Final records:", df.count())

    missing_zip = df.filter(
        F.col("postal_code_clean").isNull()
    ).count()

    invalid_dates = df.filter(
        F.col("invalid_date_order")
    ).count()

    missing_coords = df.filter(
        F.col("missing_coordinates")
    ).count()

    print(
        "Missing or invalid cleaned ZIP codes:",
        missing_zip
    )

    print(
        "Invalid date-order records:",
        invalid_dates
    )

    print(
        "Records missing coordinates:",
        missing_coords
    )

    return df


# Adjust Schemas

def adjust_nyc_schema(df):
    """
    Create common attribute names for NYC records.

    Existing NYC columns are retained.
    """

    print(
        "\n========== ADJUSTING NYC SCHEMA =========="
    )

    return (
        df
        .withColumn(
            "original_request_id",
            F.col("unique_key").cast("string")
        )
        .withColumn(
            "created_at",
            F.col("created_date").cast("timestamp")
        )
        .withColumn(
            "closed_at",
            F.col("closed_date").cast("timestamp")
        )
        .withColumn(
            "request_type",
            F.col("complaint_type").cast("string")
        )
        .withColumn(
            "status_original",
            F.col("status").cast("string")
        )
        .withColumn(
            "agency_common",
            F.col("agency").cast("string")
        )
        .withColumn(
            "postal_code",
            F.col("postal_code_clean").cast("string")
        )
        .withColumn(
            "submission_channel_original",
            F.col("open_data_channel_type").cast("string")
        )
    )


def adjust_la_schema(df):
    """
    Create common attribute names for LA records.

    Existing LA columns are retained.
    """

    print(
        "\n========== ADJUSTING LA SCHEMA =========="
    )

    return (
        df
        .withColumn(
            "original_request_id",
            F.col("srnumber").cast("string")
        )
        .withColumn(
            "created_at",
            F.col("createddate").cast("timestamp")
        )
        .withColumn(
            "closed_at",
            F.col("closeddate").cast("timestamp")
        )
        .withColumn(
            "request_type",
            F.col("requesttype").cast("string")
        )
        .withColumn(
            "status_original",
            F.col("status").cast("string")
        )
        .withColumn(
            "agency_common",
            F.col("owner").cast("string")
        )
        .withColumn(
            "postal_code",
            F.col("postal_code_clean").cast("string")
        )
        .withColumn(
            "submission_channel_original",
            F.col("requestsource").cast("string")
        )
    )


# Check common Schemas

COMMON_COLUMNS = [
    "original_request_id",
    "created_at",
    "closed_at",
    "request_type",
    "status_original",
    "agency_common",
    "postal_code",
    "submission_channel_original",
    "latitude",
    "longitude",
    "invalid_date_order",
    "missing_coordinates"
]


def validate_common_schema(nyc_df, la_df):
    """
    Ensure common columns have compatible names and types.
    """

    print(
        "\n========== COMMON SCHEMA VALIDATION =========="
    )

    nyc_common = nyc_df.select(*COMMON_COLUMNS)

    la_common = la_df.select(*COMMON_COLUMNS)

    nyc_types = dict(nyc_common.dtypes)

    la_types = dict(la_common.dtypes)

    mismatches = []

    for column in COMMON_COLUMNS:

        if nyc_types[column] != la_types[column]:

            mismatches.append(
                (
                    column,
                    nyc_types[column],
                    la_types[column]
                )
            )

    if mismatches:

        print("Schema mismatches:", mismatches)

        raise ValueError(
            "NYC and LA common schemas do not match."
        )

    print(
        "SUCCESS: All common column types match."
    )


# Standardize status and submission channels

def standardize_categories(df, city):
    """
    Create comparable status and submission-channel labels.

    Preserve source values in:
      status_original
      submission_channel_original
    """

    print(
        f"\n========== STANDARDIZING {city} =========="
    )

    status = F.col("status_original")

    channel = F.col(
        "submission_channel_original"
    )

# Standardize request statuses

    df = df.withColumn(
        "status_standardized",
        F.when(
            status == "Closed",
            "Closed"
        ).when(
            status == "Open",
            "Open"
        ).when(
            status == "Pending",
            "Pending"
        ).when(
            status.isin(
                "In Progress",
                "Assigned",
                "Started"
            ),
            "In Progress"
        ).when(
            status == "Cancelled",
            "Cancelled"
        ).when(
            status == "Referred Out",
            "Referred Out"
        ).when(
            status == "Forward",
            "Forward"
        ).when(
            status == "Unspecified",
            "Unknown"
        ).otherwise(status)
    )

# Standardize submission channels

    df = df.withColumn(
        "submission_channel_standardized",
        F.when(
            channel.isin("PHONE", "Call"),
            "Phone"
        ).when(
            channel.isin("MOBILE", "Mobile App"),
            "Mobile"
        ).when(
            channel.isin(
                "ONLINE",
                "Self Service",
                "Web Form"
            ),
            "Online / Self-service"
        ).when(
            channel == "UNKNOWN",
            "Unknown"
        ).when(
            channel == "OTHER",
            "Other"
        ).otherwise(channel)
    )

# Check standardized values

    print("\nStandardized status distribution:")

    (
        df.groupBy("status_standardized")
        .count()
        .orderBy(F.desc("count"))
        .show(30, truncate=False)
    )

    print("\nStandardized channel distribution:")

    (
        df.groupBy("submission_channel_standardized")
        .count()
        .orderBy(F.desc("count"))
        .show(30, truncate=False)
    )

    return df

# Create additional columns

def create_required_attributes(df, city):
    """
    Add:
      source_city
      global_request_id
      resolution_hours

    Invalid timestamp pairs receive NULL resolution_hours.
    """

    print(
        f"\n========== CREATING ATTRIBUTES: {city} =========="
    )

# Add source city

    df = df.withColumn(
        "source_city",
        F.lit(city)
    )

# Create unique request IDs

    df = df.withColumn(
        "global_request_id",
        F.concat(
            F.col("source_city"),
            F.lit("_"),
            F.col("original_request_id")
        )
    )

# Calculate resolution time in hours

    valid_timestamps = (
        F.col("created_at").isNotNull()
        & F.col("closed_at").isNotNull()
        & (
            F.col("closed_at")
            >= F.col("created_at")
        )
    )

    df = df.withColumn(
        "resolution_hours",
        F.when(
            valid_timestamps,
            (
                F.unix_timestamp("closed_at")
                - F.unix_timestamp("created_at")
            ) / F.lit(3600.0)
        ).otherwise(
            F.lit(None).cast("double")
        )
    )

# Validate new columns

    total_records = df.count()

    distinct_ids = (
        df.select("global_request_id")
        .distinct()
        .count()
    )

    missing_ids = df.filter(
        F.col("global_request_id").isNull()
    ).count()

    negative_hours = df.filter(
        F.col("resolution_hours") < 0
    ).count()

    null_hours = df.filter(
        F.col("resolution_hours").isNull()
    ).count()

    print("\nATTRIBUTE VALIDATION:")

    print("Total records:", total_records)

    print("Distinct global IDs:", distinct_ids)

    print("Missing global IDs:", missing_ids)

    print("Negative resolution hours:", negative_hours)

    print("NULL resolution hours:", null_hours)

    if (
        distinct_ids != total_records
        or missing_ids != 0
        or negative_hours != 0
    ):
        raise ValueError(
            f"{city}: Attribute validation failed."
        )

    print(
        f"SUCCESS: {city} attributes validated."
    )

    return df


# Save and validate cleaned datasets

def persist_and_validate(df, city, output_name):
    """
    Save a processed dataset to Parquet.

    Reload the files and verify:
    - Record count
    - Column names
    - Column data types
    - Source city
    - Global identifiers

    Print detailed schema differences when detected.
    """

    print(
        f"\n========== SAVING {city} DATASET =========="
    )

    output_path = OUTPUT_DIR / output_name

    expected_count = df.count()

# Save cleaned data as Parquet

    df.write.mode("overwrite").parquet(
        str(output_path)
    )

    print(f"Saved to: {output_path}")

# Reload saved data

    reloaded_df = spark.read.parquet(
        str(output_path)
    )

    actual_count = reloaded_df.count()

# Check schema consistency

# Compare column names and types, ignoring order and nullability

    original_types = dict(df.dtypes)

    reloaded_types = dict(reloaded_df.dtypes)

    schema_matches = (
        original_types == reloaded_types
    )

    print("\nSCHEMA COMPARISON:")

    all_columns = sorted(
        set(original_types) | set(reloaded_types)
    )

    differences_found = False

    for column in all_columns:

        original_type = original_types.get(column)

        saved_type = reloaded_types.get(column)

        if original_type != saved_type:

            differences_found = True

            print(
                f"Column: {column}"
            )

            print(
                f"  Before saving: {original_type}"
            )

            print(
                f"  After loading: {saved_type}"
            )

    if not differences_found:

        print(
            "No column name or data type differences found."
        )

# Validate city labels and request IDs

    invalid_ids = reloaded_df.filter(
        F.col("global_request_id").isNull()
        | (
            ~F.col("global_request_id").startswith(
                city + "_"
            )
        )
    ).count()

    incorrect_city = reloaded_df.filter(
        F.col("source_city").isNull()
        | (F.col("source_city") != city)
    ).count()

# Print validation results

    print("\nPERSISTENCE VALIDATION:")

    print(
        "Expected records:",
        expected_count
    )

    print(
        "Reloaded records:",
        actual_count
    )

    print(
        "Schema matches:",
        schema_matches
    )

    print(
        "Invalid or missing global IDs:",
        invalid_ids
    )

    print(
        "Incorrect source-city values:",
        incorrect_city
    )

# Check for validation errors

    if (
        actual_count != expected_count
        or not schema_matches
        or invalid_ids != 0
        or incorrect_city != 0
    ):

        raise ValueError(
            f"{city}: Persistence validation failed. "
            "Review the differences displayed above."
        )

# Display sample records

    print("\nReloaded sample records:")

    reloaded_df.select(
        "source_city",
        "global_request_id",
        "request_type",
        "status_standardized",
        "submission_channel_standardized",
        "resolution_hours"
    ).show(5, truncate=False)

    print(
        f"\nSUCCESS: {city} data saved and verified."
    )

    return reloaded_df


# Run data processing

try:

# Load NYC and LA datasets

    nyc_df = spark.read.csv(
        str(DATA_DIR / "nyc_311_800k.csv"),
        header=True,
        inferSchema=True
    )

    la_df = spark.read.csv(
        str(DATA_DIR / "la_311_800k.csv"),
        header=True,
        inferSchema=True
    )

# Clean both datasets

    nyc_clean = clean_dataset(
        nyc_df,
        "NYC",
        "incident_zip",
        "created_date",
        "closed_date"
    )

    la_clean = clean_dataset(
        la_df,
        "LA",
        "zipcode",
        "createddate",
        "closeddate"
    )

# Map columns to common schema

    nyc_mapped = adjust_nyc_schema(
        nyc_clean
    )

    la_mapped = adjust_la_schema(
        la_clean
    )

    validate_common_schema(
        nyc_mapped,
        la_mapped
    )

# Standardize categorical values

    nyc_standardized = standardize_categories(
        nyc_mapped,
        "NYC"
    )

    la_standardized = standardize_categories(
        la_mapped,
        "LA"
    )

# Add derived columns

    nyc_processed = create_required_attributes(
        nyc_standardized,
        "NYC"
    )

    la_processed = create_required_attributes(
        la_standardized,
        "LA"
    )

# Save cleaned datasets

    nyc_saved = persist_and_validate(
        nyc_processed,
        "NYC",
        "nyc_cleaned"
    )

    la_saved = persist_and_validate(
        la_processed,
        "LA",
        "la_cleaned"
    )

    print("\nData processing completed successfully.")

    print(
        "NYC and LA processed datasets saved successfully."
    )

    print(
        "Both datasets passed persistence validation."
    )

    print(
        "Original CSV files remain unchanged."
    )

finally:

    spark.stop()
