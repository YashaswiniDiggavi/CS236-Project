
import os
from pathlib import Path

from pyspark.sql import SparkSession

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA = PROJECT_ROOT / "processed_data"

# PostgreSQL connection configuration
JDBC_URL = "jdbc:postgresql://127.0.0.1:5432/cs236_project"
DB_USER = "cs236"
DB_PASSWORD = os.environ.get("CS236_DB_PASSWORD")

if not DB_PASSWORD:
    raise RuntimeError(
        "Set CS236_DB_PASSWORD before running this script."
    )

# Existing tables and their corresponding Phase 1 datasets
DATASETS = [
    ("nyc_cleaned", 800000),
    ("la_cleaned", 800000),
    ("merged_311", 1600000),
]

spark = (
    SparkSession.builder
    .appName("CS236_Phase2_PostgreSQL_Population")
    .master("local[4]")
    .config("spark.sql.session.timeZone", "UTC")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

JDBC_OPTIONS = {
    "url": JDBC_URL,
    "user": DB_USER,
    "password": DB_PASSWORD,
    "driver": "org.postgresql.Driver",
}

try:
    for table_name, expected_rows in DATASETS:
        print(f"\n========== {table_name} ==========")

        # Load the existing cleaned or merged Phase 1 dataset.
        parquet_path = str(PROCESSED_DATA / table_name)
        df = spark.read.parquet(parquet_path)

        actual_rows = df.count()
        print("Spark row count:", actual_rows)
        print("Number of columns:", len(df.columns))

        if actual_rows != expected_rows:
            raise ValueError(
                f"{table_name}: Expected {expected_rows} rows, "
                f"found {actual_rows}"
            )

        # Check the existing PostgreSQL table before inserting.
        current_rows = (
            spark.read.format("jdbc")
            .options(**JDBC_OPTIONS)
            .option(
                "dbtable",
                f"(SELECT COUNT(*) AS row_count "
                f"FROM {table_name}) AS existing"
            )
            .load()
            .first()["row_count"]
        )

        if current_rows != 0:
            raise RuntimeError(
                f"{table_name} already contains {current_rows} rows. "
                "Stopping to prevent duplicate or repeated loading."
            )

        print("PostgreSQL table is empty.")
        print("Loading records using PySpark JDBC...")

        # Append to the tables created by create_tables.sql.
        # Limit parallel JDBC connections to four.
        (
            df.coalesce(4)
            .write
            .format("jdbc")
            .options(**JDBC_OPTIONS)
            .option("dbtable", table_name)
            .option("batchsize", "5000")
            .mode("append")
            .save()
        )

        # Confirm the inserted row count using PostgreSQL.
        loaded_rows = (
            spark.read.format("jdbc")
            .options(**JDBC_OPTIONS)
            .option(
                "dbtable",
                f"(SELECT COUNT(*) AS row_count "
                f"FROM {table_name}) AS loaded"
            )
            .load()
            .first()["row_count"]
        )

        print("PostgreSQL row count:", loaded_rows)

        if loaded_rows != expected_rows:
            raise RuntimeError(
                f"{table_name}: PostgreSQL count mismatch. "
                f"Expected {expected_rows}, got {loaded_rows}."
            )

        print(f"SUCCESS: {table_name} loaded and count verified.")

    print("\nALL THREE DATASETS LOADED SUCCESSFULLY.")
    print("Expected total records: 3,200,000")

finally:
    spark.stop()
