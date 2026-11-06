
from pathlib import Path
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = str(PROJECT_ROOT / "processed_data" / "merged_311")
OUTPUT_DIR = PROJECT_ROOT / "phase2" / "analysis_results"

spark = (
    SparkSession.builder
    .appName("CS236_Phase2_Status_Channel_Analysis")
    .master("local[4]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    # Load the Phase 1 unified dataset.
    df = spark.read.parquet(INPUT_PATH)

    assert df.count() == 1600000, "Unexpected dataset count"

    # Denominator: all service requests belonging to each city.
    city_totals = (
        df.groupBy("source_city")
        .agg(F.count("*").alias("city_total"))
    )

    def analyze_category(column_name, label, output_folder):
        # Label missing standardized values for reporting only.
        data = df.withColumn(
            "category",
            F.coalesce(
                F.col(column_name),
                F.lit("Unknown")
            )
        )

        # Count requests in each standardized category.
        counts = (
            data.groupBy("source_city", "category")
            .agg(F.count("*").alias("request_count"))
            .join(city_totals, on="source_city")
            .withColumn(
                "percentage",
                F.round(
                    F.col("request_count") * 100.0
                    / F.col("city_total"),
                    2
                )
            )
        )

        # Rank categories within each city by request volume.
        rank_window = Window.partitionBy(
            "source_city"
        ).orderBy(
            F.desc("request_count"),
            F.asc("category")
        )

        results = (
            counts
            .withColumn(
                "rank",
                F.row_number().over(rank_window)
            )
            .select(
                "source_city",
                "rank",
                "category",
                "request_count",
                "city_total",
                "percentage"
            )
        )

        print(f"\n=== {label.upper()} DISTRIBUTION ===")
        results.orderBy(
            "source_city", "rank"
        ).show(100, truncate=False)

        print(f"\n=== MOST COMMON {label.upper()} ===")
        results.filter(F.col("rank") == 1).show(
            truncate=False
        )

        # Validate that all requests were counted.
        totals = (
            results.groupBy("source_city")
            .agg(
                F.sum("request_count").alias("counted")
            )
            .join(city_totals, on="source_city")
        )

        mismatches = totals.filter(
            F.col("counted") != F.col("city_total")
        ).count()

        assert mismatches == 0, (
            f"{label}: category counts do not match city totals"
        )

        # Save results as CSV for the report.
        (
            results.orderBy("source_city", "rank")
            .coalesce(1)
            .write.mode("overwrite")
            .option("header", True)
            .csv(str(OUTPUT_DIR / output_folder))
        )

        print(f"VALIDATION PASSED: {label}")
        return results

    status_results = analyze_category(
        "status_standardized",
        "Status",
        "status_distribution"
    )

    channel_results = analyze_category(
        "submission_channel_standardized",
        "Submission Channel",
        "channel_distribution"
    )

    print("\nSUCCESS: Both bonus analyses completed.")

finally:
    spark.stop()
