from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed_data"

NYC_PATH = PROCESSED_DIR / "nyc_cleaned"
LA_PATH = PROCESSED_DIR / "la_cleaned"
MERGED_PATH = PROCESSED_DIR / "merged_311"


# Common schema for both cities

COMMON_COLUMNS = [
    "source_city",
    "global_request_id",
    "original_request_id",
    "created_at",
    "closed_at",
    "request_type",
    "status_original",
    "status_standardized",
    "agency_common",
    "postal_code",
    "postal_code_original",
    "suspicious_postal_code",
    "submission_channel_original",
    "submission_channel_standardized",
    "latitude",
    "longitude",
    "invalid_date_order",
    "missing_coordinates",
    "invalid_coordinates",
    "resolution_hours"
]


# Spark setup

spark = (
    SparkSession.builder
    .appName("CS236_Phase1_Merge")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.default.parallelism", "4")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# Validate common schema

def check_schema(nyc, la):

    for city, df in [
        ("NYC", nyc),
        ("LA", la)
    ]:

        missing = set(COMMON_COLUMNS) - set(df.columns)

        if missing:
            raise ValueError(
                f"{city} missing columns: {sorted(missing)}"
            )

    nyc_types = dict(nyc.dtypes)
    la_types = dict(la.dtypes)

    differences = [
        (
            column,
            nyc_types[column],
            la_types[column]
        )
        for column in COMMON_COLUMNS
        if nyc_types[column] != la_types[column]
    ]

    if differences:
        raise ValueError(
            f"Mismatched column types: {differences}"
        )

    print("SUCCESS: Common schema validation passed.")


# Validate merged dataset

def validate(df, expected_city_counts):

    expected = sum(expected_city_counts.values())
    actual = df.count()

    print("\nTotal merged records:", actual)
    print("Expected merged records:", expected)

    if actual != expected:
        raise ValueError(
            "Merged record count mismatch"
        )

    # Check records for each city

    counts = {
        row["source_city"]: row["count"]
        for row in (
            df.groupBy("source_city")
            .count()
            .collect()
        )
    }

    print("Source-city counts:", counts)

    if counts != expected_city_counts:
        raise ValueError(
            "Source-city count mismatch"
        )

    # Check request identifiers

    distinct_ids = df.select(
        "global_request_id"
    ).distinct().count()

    missing_ids = df.filter(
        F.col("global_request_id").isNull()
    ).count()

    bad_prefixes = df.filter(
        (~F.col("global_request_id").startswith("NYC_"))
        & (~F.col("global_request_id").startswith("LA_"))
    ).count()

    wrong_city_prefixes = df.filter(
        (
            (F.col("source_city") == "NYC")
            & (~F.col("global_request_id").startswith("NYC_"))
        )
        |
        (
            (F.col("source_city") == "LA")
            & (~F.col("global_request_id").startswith("LA_"))
        )
    ).count()

    print("Distinct global IDs:", distinct_ids)
    print("Missing global IDs:", missing_ids)
    print(
        "Incorrect ID prefixes:",
        bad_prefixes + wrong_city_prefixes
    )

    # Check resolution hours

    negative_hours = df.filter(
        F.col("resolution_hours") < 0
    ).count()

    null_hours = df.filter(
        F.col("resolution_hours").isNull()
    ).count()

    print("Negative resolution hours:", negative_hours)
    print("NULL resolution hours:", null_hours)

    # Check essential fields

    essential_columns = [
        "source_city",
        "global_request_id",
        "original_request_id",
        "created_at",
        "request_type"
    ]

    for column in essential_columns:

        missing = df.filter(
            F.col(column).isNull()
        ).count()

        print(f"Missing {column}: {missing}")

        if missing:
            raise ValueError(
                f"Missing essential values: {column}"
            )

    if (
        distinct_ids != actual
        or missing_ids
        or bad_prefixes
        or wrong_city_prefixes
        or negative_hours
    ):
        raise ValueError(
            "Merged dataset validation failed"
        )

    print("SUCCESS: Merged validation passed.")


# Merge NYC and LA datasets

try:

    print("\nLoading cleaned datasets")

    nyc_df = spark.read.parquet(str(NYC_PATH))
    la_df = spark.read.parquet(str(LA_PATH))

    nyc_count = nyc_df.count()
    la_count = la_df.count()

    print("NYC records:", nyc_count)
    print("LA records:", la_count)

    # Check columns and types

    check_schema(nyc_df, la_df)

    # Select matching fields and merge

    merged = (
        nyc_df.select(*COMMON_COLUMNS)
        .unionByName(
            la_df.select(*COMMON_COLUMNS)
        )
    )

    print("\nMerged dataset schema:")
    merged.printSchema()

    expected_counts = {
        "NYC": nyc_count,
        "LA": la_count
    }

    # Validate before saving

    validate(merged, expected_counts)

    # Save merged dataset

    print("\nSaving merged dataset")

    merged.write.mode("overwrite").parquet(
        str(MERGED_PATH)
    )

    # Reload and validate saved data

    print("\nReloading merged dataset")

    reloaded = spark.read.parquet(
        str(MERGED_PATH)
    )

    if dict(reloaded.dtypes) != dict(merged.dtypes):
        raise ValueError(
            "Reloaded merged schema differs"
        )

    validate(reloaded, expected_counts)

    print(
        "\nSUCCESS: Merged dataset saved and verified."
    )

finally:

    spark.stop()
