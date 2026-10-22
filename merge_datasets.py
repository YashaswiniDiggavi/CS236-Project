
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Merge and validate NYC and LA service-request datasets

BASE_DIR = Path(__file__).resolve().parent

PROCESSED_DIR = BASE_DIR / "processed_data"

NYC_PATH = PROCESSED_DIR / "nyc_cleaned"
LA_PATH = PROCESSED_DIR / "la_cleaned"

MERGED_PATH = PROCESSED_DIR / "merged_311"


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

heap_gb = (
    spark.sparkContext._jvm.java.lang.Runtime
    .getRuntime()
    .maxMemory()
    / (1024 ** 3)
)

print(f"\nSpark JVM maximum heap: {heap_gb:.2f} GB")

# Common columns

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
    "submission_channel_original",
    "submission_channel_standardized",
    "latitude",
    "longitude",
    "invalid_date_order",
    "missing_coordinates",
    "resolution_hours"
]


def verify_columns(df, city):
    """
    Verify that all required common columns exist.
    """

    missing = [
        column
        for column in COMMON_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{city}: Missing required columns: {missing}"
        )

    print(
        f"SUCCESS: {city} contains all required columns."
    )


def validate_merged(df):
    """
    Validate the unified dataset.

    Checks:
      1. Total records
      2. City distribution
      3. Global ID uniqueness
      4. Missing IDs
      5. Incorrect city prefixes
      6. Negative resolution hours
      7. Missing essential fields
    """

    print(
        "\n========== STEP 7: FINAL VALIDATION =========="
    )

# Load cleaned datasets

    total = df.count()

    print("\nTotal merged records:", total)

    if total != 1_600_000:
        raise ValueError(
            f"Expected 1,600,000 rows, found {total}"
        )

# Check records by city

    print("\nRecords by source city:")

    df.groupBy("source_city").count().show()

    city_counts = {
        row["source_city"]: row["count"]
        for row in df.groupBy("source_city").count().collect()
    }

    if city_counts != {
        "NYC": 800_000,
        "LA": 800_000
    }:
        raise ValueError(
            f"Unexpected source-city counts: {city_counts}"
        )

# Check unique IDs

    unique_ids = (
        df.select("global_request_id")
        .distinct()
        .count()
    )

    missing_ids = df.filter(
        F.col("global_request_id").isNull()
        | (F.trim("global_request_id") == "")
    ).count()

    print("Distinct global IDs:", unique_ids)
    print("Missing global IDs:", missing_ids)

    if unique_ids != total or missing_ids != 0:
        raise ValueError(
            "Global request IDs are not unique and complete."
        )
    
# Check ID prefixes

    invalid_prefixes = df.filter(
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

    print("Incorrect ID prefixes:", invalid_prefixes)

    if invalid_prefixes != 0:
        raise ValueError(
            "Some global IDs have incorrect city prefixes."
        )

# Check resolution times

    negative_resolution = df.filter(
        F.col("resolution_hours") < 0
    ).count()

    missing_resolution = df.filter(
        F.col("resolution_hours").isNull()
    ).count()

    print(
        "Negative resolution hours:",
        negative_resolution
    )

    print(
        "NULL resolution hours:",
        missing_resolution
    )

    if negative_resolution != 0:
        raise ValueError(
            "Negative resolution hours found."
        )

# Check missing values

    print("\nMissing essential field counts:")

    essential_columns = [
        "source_city",
        "global_request_id",
        "original_request_id",
        "created_at",
        "request_type"
    ]

    for column in essential_columns:
        missing_count = df.filter(
            F.col(column).isNull()
        ).count()

        print(f"{column}: {missing_count}")

# Missing values are reported rather than silently
# dropping records from the source datasets.

    print("\nSUCCESS: Final merged validation passed.")


try:

# Load cleaned datasets

    print(
        "\n========== STEP 6: LOADING CLEANED DATASETS =========="
    )

    if not NYC_PATH.exists():
        raise FileNotFoundError(
            f"NYC dataset not found: {NYC_PATH}"
        )

    if not LA_PATH.exists():
        raise FileNotFoundError(
            f"LA dataset not found: {LA_PATH}"
        )

    nyc_df = spark.read.parquet(str(NYC_PATH))
    la_df = spark.read.parquet(str(LA_PATH))

    nyc_count = nyc_df.count()
    la_count = la_df.count()

    print("NYC records:", nyc_count)
    print("LA records:", la_count)

    if nyc_count != 800_000 or la_count != 800_000:
        raise ValueError(
            "Source dataset counts are incorrect."
        )

# Check common schema

    print(
        "\n========== VERIFYING COMMON SCHEMA =========="
    )

    verify_columns(nyc_df, "NYC")
    verify_columns(la_df, "LA")

    # Select consistent fields and ordering.
    nyc_common = nyc_df.select(*COMMON_COLUMNS)
    la_common = la_df.select(*COMMON_COLUMNS)

    nyc_types = dict(nyc_common.dtypes)
    la_types = dict(la_common.dtypes)

    for column in COMMON_COLUMNS:
        if nyc_types[column] != la_types[column]:
            raise ValueError(
                f"Data type mismatch in {column}: "
                f"NYC={nyc_types[column]}, "
                f"LA={la_types[column]}"
            )

    print("SUCCESS: Common column data types match.")

# Merge datasets

    print(
        "\n========== MERGING NYC AND LA =========="
    )

    merged_df = nyc_common.unionByName(
        la_common
    )

    merged_count = merged_df.count()

    print("Merged record count:", merged_count)

    print("\nMerged dataset schema:")

    merged_df.printSchema()

    print("\nSample merged records:")

    merged_df.select(
        "source_city",
        "global_request_id",
        "request_type",
        "status_standardized",
        "resolution_hours"
    ).show(10, truncate=False)

# Validate merged data

    validate_merged(merged_df)

# Save merged data

    print(
        "\n========== SAVING MERGED DATASET =========="
    )

    merged_df.write.mode("overwrite").parquet(
        str(MERGED_PATH)
    )

    print("Saved to:", MERGED_PATH)

# Verify saved data

    print(
        "\n========== RELOADING MERGED DATASET =========="
    )

    reloaded_df = spark.read.parquet(
        str(MERGED_PATH)
    )

    saved_count = reloaded_df.count()

    schema_matches = (
        dict(merged_df.dtypes)
        == dict(reloaded_df.dtypes)
    )

    saved_city_counts = {
        row["source_city"]: row["count"]
        for row in reloaded_df.groupBy(
            "source_city"
        ).count().collect()
    }

    saved_unique_ids = (
        reloaded_df
        .select("global_request_id")
        .distinct()
        .count()
    )

    print("Reloaded records:", saved_count)
    print("Schema matches:", schema_matches)
    print("Source-city counts:", saved_city_counts)
    print("Distinct saved IDs:", saved_unique_ids)

    if (
        saved_count != 1_600_000
        or not schema_matches
        or saved_unique_ids != saved_count
        or saved_city_counts != {
            "NYC": 800_000,
            "LA": 800_000
        }
    ):
        raise ValueError(
            "Saved merged dataset failed validation."
        )

    print(
        "\n========== PHASE 1 DATA PROCESSING COMPLETED =========="
    )

    print(
        "SUCCESS: 1,600,000 NYC and LA records merged."
    )

    print(
        "SUCCESS: Merged data saved and reloaded."
    )

    print(
        "SUCCESS: All required validation checks passed."
    )

finally:

    spark.stop()
