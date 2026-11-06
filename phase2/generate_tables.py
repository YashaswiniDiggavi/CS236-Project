from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StringType, TimestampType, DoubleType,
    FloatType, IntegerType, LongType,
    BooleanType, DateType, DecimalType
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE2_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = PHASE2_DIR / "create_tables.sql"

DATASETS = {
    "nyc_cleaned": "nyc_cleaned",
    "la_cleaned": "la_cleaned",
    "merged_311": "merged_311",
}

# Columns verified to contain no NULL values in all three datasets.
REQUIRED_COLUMNS = {
    "global_request_id",
    "source_city",
    "created_at",
    "request_type",
    "is_valid_completed",
}


def sql_type(spark_type):
    """Translate a Spark data type into a PostgreSQL data type."""
    if isinstance(spark_type, StringType):
        return "TEXT"
    if isinstance(spark_type, TimestampType):
        return "TIMESTAMP"
    if isinstance(spark_type, DoubleType):
        return "DOUBLE PRECISION"
    if isinstance(spark_type, FloatType):
        return "REAL"
    if isinstance(spark_type, IntegerType):
        return "INTEGER"
    if isinstance(spark_type, LongType):
        return "BIGINT"
    if isinstance(spark_type, BooleanType):
        return "BOOLEAN"
    if isinstance(spark_type, DateType):
        return "DATE"
    if isinstance(spark_type, DecimalType):
        return f"NUMERIC({spark_type.precision}, {spark_type.scale})"

    raise ValueError(f"Unsupported Spark type: {spark_type}")


def quote_identifier(name):
    """Safely quote a SQL identifier."""
    return '"' + name.replace('"', '""') + '"'


spark = (
    SparkSession.builder
    .appName("CS236_Phase2_Generate_PostgreSQL_Tables")
    .master("local[4]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    statements = [
        "-- CS236 Phase 2: PostgreSQL table definitions",
        "-- Generated from the actual Phase 1 Parquet schemas.",
        "-- Required columns use NOT NULL constraints.",
        "-- Optional columns remain nullable.",
        "",
    ]

    for table_name, folder in DATASETS.items():
        path = str(PROJECT_ROOT / "processed_data" / folder)
        df = spark.read.parquet(path)

        print(f"\nProcessing: {table_name}")
        print(f"Number of columns: {len(df.columns)}")

        # Verify all required columns exist in the dataset.
        missing_columns = REQUIRED_COLUMNS - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"Required columns missing from {table_name}: "
                f"{sorted(missing_columns)}"
            )

        column_definitions = []

        for field in df.schema.fields:
            column_name = quote_identifier(field.name)
            column_type = sql_type(field.dataType)

            # Apply NOT NULL to mandatory attributes.
            if field.name in REQUIRED_COLUMNS:
                definition = f"    {column_name} {column_type} NOT NULL"
            else:
                definition = f"    {column_name} {column_type}"

            column_definitions.append(definition)

        # Global request ID uniquely identifies each record.
        column_definitions.append(
            '    PRIMARY KEY ("global_request_id")'
        )

        statement = (
            f"CREATE TABLE IF NOT EXISTS "
            f"{quote_identifier(table_name)} (\n"
            + ",\n".join(column_definitions)
            + "\n);"
        )

        statements.append(statement)
        statements.append("")

    OUTPUT_FILE.write_text(
        "\n".join(statements),
        encoding="utf-8"
    )

    print("\nSUCCESS: PostgreSQL SQL file generated.")
    print("File:", OUTPUT_FILE)

finally:
    spark.stop()
