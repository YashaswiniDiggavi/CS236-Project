from pathlib import Path
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = str(PROJECT_ROOT / "processed_data" / "merged_311")
RESULTS_DIR = PROJECT_ROOT / "phase2" / "analysis_results"

spark = (
    SparkSession.builder
    .appName("CS236_Phase2_ZIP_Hotspots")
    .master("local[4]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    # Read the unified dataset from Phase 1.
    df = spark.read.parquet(INPUT_PATH)

    # Use standardized, nonmissing five-digit postal codes.
    requests = (
        df.select("source_city", "postal_code", "request_type")
        .filter(
            F.col("source_city").isNotNull()
            & F.col("postal_code").rlike(r"^[0-9]{5}$")
        )
    )

    print("\n=== RECORD VALIDATION ===")
    print("Total unified records:", df.count())
    print("Records with usable ZIP codes:", requests.count())

    # Count requests by ZIP code and city.
    zip_counts = (
        requests.groupBy("source_city", "postal_code")
        .agg(F.count("*").alias("total_requests"))
    )

    # Select exactly 10 ZIP codes per city.
    zip_window = Window.partitionBy("source_city").orderBy(
        F.desc("total_requests"),
        F.asc("postal_code")
    )

    top10 = (
        zip_counts
        .withColumn("zip_rank", F.row_number().over(zip_window))
        .filter(F.col("zip_rank") <= 10)
    )

    print("\n=== TOP 10 ZIP CODES PER CITY ===")
    top10.orderBy(
        "source_city", "zip_rank"
    ).show(20, truncate=False)

    # Count every request type in each ZIP code.
    # Missing types are labeled Unknown rather than discarded.
    type_counts = (
        requests.withColumn(
            "request_type",
            F.coalesce(F.col("request_type"), F.lit("Unknown"))
        )
        .groupBy("source_city", "postal_code", "request_type")
        .agg(F.count("*").alias("type_count"))
    )

    # Keep only request types within the top 10 ZIP codes.
    top_types = type_counts.join(
        top10,
        on=["source_city", "postal_code"],
        how="inner"
    )

    # Rank the three most frequent request types in each ZIP.
    type_window = Window.partitionBy(
        "source_city", "postal_code"
    ).orderBy(
        F.desc("type_count"),
        F.asc("request_type")
    )

    top3 = (
        top_types
        .withColumn("type_rank", F.row_number().over(type_window))
        .filter(F.col("type_rank") <= 3)
        .withColumn(
            "percentage",
            F.round(
                F.col("type_count") * 100.0 / F.col("total_requests"),
                2
            )
        )
        .select(
            "source_city",
            "zip_rank",
            "postal_code",
            "total_requests",
            "type_rank",
            "request_type",
            "type_count",
            "percentage"
        )
        .orderBy("source_city", "zip_rank", "type_rank")
    )

    print("\n=== TOP 3 REQUEST TYPES IN EACH HOTSPOT ===")
    top3.show(60, truncate=False)

    # Ensure we found 10 ZIP codes in each city.
    print("\n=== VALIDATION ===")
    top10.groupBy("source_city").count().show()

    assert top10.count() == 20, "Expected 20 hotspot ZIP codes"
    assert top3.count() == 60, "Expected 60 ranked request types"

    invalid_percentages = top3.filter(
        (F.col("percentage") < 0) |
        (F.col("percentage") > 100)
    ).count()
    assert invalid_percentages == 0, "Invalid percentages found"

    print("VALIDATION PASSED: 20 ZIP codes and 60 request types.")

    # Save both result tables for the report.
    (
        top10.orderBy("source_city", "zip_rank")
        .coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv(str(RESULTS_DIR / "top10_zipcodes"))
    )

    (
        top3.coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv(str(RESULTS_DIR / "top3_request_types"))
    )

    print("\nSUCCESS: Hotspot analysis results saved.")

finally:
    spark.stop()