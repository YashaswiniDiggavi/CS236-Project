
from pathlib import Path

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = str(PROJECT_ROOT / "processed_data" / "merged_311")
OUTPUT_DIR = PROJECT_ROOT / "phase2" / "analysis_results"

spark = (
    SparkSession.builder
    .appName("CS236_Phase2_Resolution_Performance")
    .master("local[4]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    df = spark.read.parquet(INPUT_PATH)

    # Only completed requests with valid, nonnegative durations.
    completed = df.filter(
        (F.col("is_valid_completed") == True)
        & F.col("resolution_hours").isNotNull()
        & (F.col("resolution_hours") >= 0)
    )

    print("\n=== VALID COMPLETED REQUESTS ===")
    completed.groupBy("source_city").count().show()

    # Average and approximate median resolution time per city.
    city_performance = (
        completed.groupBy("source_city")
        .agg(
            F.count("*").alias("completed_requests"),
            F.round(
                F.avg("resolution_hours"), 2
            ).alias("avg_resolution_hours"),
            F.round(
                F.percentile_approx(
                    "resolution_hours", 0.5, 10000
                ), 2
            ).alias("median_resolution_hours")
        )
        .orderBy("source_city")
    )

    print("\n=== RESOLUTION PERFORMANCE BY CITY ===")
    city_performance.show(truncate=False)

    # Average resolution time for each request type.
    type_performance = (
        completed
        .filter(F.col("request_type").isNotNull())
        .groupBy("source_city", "request_type")
        .agg(
            F.count("*").alias("completed_requests"),
            F.round(
                F.avg("resolution_hours"), 2
            ).alias("avg_resolution_hours")
        )
    )

    # Find five request types with the longest average duration
    # in each city. Break ties by request-type name.
    ranking = Window.partitionBy("source_city").orderBy(
        F.desc("avg_resolution_hours"),
        F.asc("request_type")
    )

    top5_slowest = (
        type_performance
        .withColumn("rank", F.row_number().over(ranking))
        .filter(F.col("rank") <= 5)
        .orderBy("source_city", "rank")
    )

    print("\n=== FIVE SLOWEST REQUEST TYPES PER CITY ===")
    top5_slowest.show(10, truncate=False)

    # Verify the Phase 1 valid-completed counts.
    assert completed.count() == 1432311, (
        "Valid-completed count differs from Phase 1"
    )

    assert top5_slowest.count() == 10, (
        "Expected five request types per city"
    )

    print("\nVALIDATION PASSED")

    # Save both result tables for the report.
    (
        city_performance.coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv(str(OUTPUT_DIR / "resolution_by_city"))
    )

    (
        top5_slowest.coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv(str(OUTPUT_DIR / "slowest_request_types"))
    )

    print("SUCCESS: Resolution analysis results saved.")

finally:
    spark.stop()
