
from pathlib import Path
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

# Locate the project root so the script works from the phase2 folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = str(PROJECT_ROOT / "processed_data" / "merged_311")
OUTPUT_PATH = str(PROJECT_ROOT / "phase2" / "analysis_results" / "weekly_requests")

spark = (
    SparkSession.builder
    .appName("CS236_Phase2_Weekly_Request_Volume")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    # Load the unified dataset created in Phase 1.
    df = spark.read.parquet(INPUT_PATH)

    print("\n=== UNIFIED DATASET ===")
    print("Total records:", df.count())
    print("Columns:", df.columns)

    # Check that the required attributes exist.
    required = {"source_city", "created_at"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Analyze submission dates. Each month has four assignment-defined
    # periods rather than calendar/ISO weeks.
    requests = (
        df.select(
            "source_city",
            F.to_date("created_at").alias("request_date")
        )
        .filter(
            F.col("source_city").isNotNull()
            & F.col("request_date").isNotNull()
        )
        .withColumn("year", F.year("request_date"))
        .withColumn("month", F.month("request_date"))
        .withColumn("day", F.dayofmonth("request_date"))
        .withColumn(
            "period",
            F.when(F.col("day") <= 7, 1)
             .when(F.col("day") <= 14, 2)
             .when(F.col("day") <= 21, 3)
             .otherwise(4)
        )
    )

    print("\nRecords with usable city/date:", requests.count())

    # Count service requests in each period for each city and month.
    weekly_counts = (
        requests.groupBy("source_city", "year", "month", "period")
        .agg(F.count("*").alias("request_count"))
        .orderBy("source_city", "year", "month", "period")
    )

    print("\n=== REQUEST VOLUMES BY MONTH AND PERIOD ===")
    weekly_counts.show(100, truncate=False)

    # Highest and lowest periods for each city, across all months.
    # Dense ranking keeps all tied periods.
    highest_window = Window.partitionBy("source_city").orderBy(
        F.desc("request_count")
    )
    lowest_window = Window.partitionBy("source_city").orderBy(
        F.asc("request_count")
    )

    ranked = (
        weekly_counts
        .withColumn("highest_rank", F.dense_rank().over(highest_window))
        .withColumn("lowest_rank", F.dense_rank().over(lowest_window))
    )

    print("\n=== HIGHEST-VOLUME PERIOD(S) PER CITY ===")
    ranked.filter(F.col("highest_rank") == 1).select(
        "source_city", "year", "month", "period", "request_count"
    ).show(truncate=False)

    print("\n=== LOWEST-VOLUME PERIOD(S) PER CITY ===")
    ranked.filter(F.col("lowest_rank") == 1).select(
        "source_city", "year", "month", "period", "request_count"
    ).show(truncate=False)

    # The period counts should sum back to the number of usable records.
    total_grouped = weekly_counts.agg(
        F.sum("request_count").alias("total")
    ).first()["total"]

    total_usable = requests.count()
    assert total_grouped == total_usable, "Count validation failed"

    print("\nVALIDATION PASSED: Period counts match usable records.")

    # Save the results for the report and presentation.
    (
        weekly_counts.coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv(OUTPUT_PATH)
    )

    print("Saved weekly analysis to:", OUTPUT_PATH)

finally:
    spark.stop()
