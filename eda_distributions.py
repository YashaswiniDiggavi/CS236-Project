from pathlib import Path
import csv

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# Exploratory data analysis

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "eda_results"

OUTPUT_DIR.mkdir(exist_ok=True)


# Spark setup

spark = (
    SparkSession.builder
    .appName("CS236_Phase1_Distributions")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.default.parallelism", "4")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# Save summary results

def save_results(df, filename):
    """
    Save aggregated Spark results to a CSV file.
    """

    rows = df.collect()
    path = OUTPUT_DIR / filename

    with open(
        path, "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.writer(file)
        writer.writerow(df.columns)

        for row in rows:
            writer.writerow(list(row))

    print(f"Saved: {path}")


# Categorical distributions

def category_analysis(df, column, name, label):

    print(f"\n{name}: {label}")

    result = (
        df.groupBy(column)
        .count()
        .orderBy(F.desc("count"))
    )

    print("Distinct categories:", result.count())

    result.show(10, truncate=False)

    save_results(
        result.limit(50),
        f"{name.lower()}_{label}.csv"
    )


# Monthly request trends

def monthly_analysis(df, date_column, name):

    print(f"\n{name}: Monthly requests")

    result = (
        df.withColumn(
            "month",
            F.date_format(
                F.col(date_column), "yyyy-MM"
            )
        )
        .groupBy("month")
        .count()
        .orderBy("month")
    )

    result.show(100, truncate=False)

    save_results(
        result,
        f"{name.lower()}_monthly.csv"
    )


# Geographic distributions

def geographic_analysis(df, zip_column, name):

    print(f"\n{name}: Top postal codes")

    result = (
        df.groupBy(zip_column)
        .count()
        .orderBy(F.desc("count"))
    )

    result.show(10, truncate=False)

    save_results(
        result.limit(20),
        f"{name.lower()}_postal_codes.csv"
    )

    print(f"\n{name}: Coordinate summary")

    df.select(
        F.min("latitude").alias(
            "minimum_latitude"
        ),
        F.max("latitude").alias(
            "maximum_latitude"
        ),
        F.avg("latitude").alias(
            "average_latitude"
        ),
        F.min("longitude").alias(
            "minimum_longitude"
        ),
        F.max("longitude").alias(
            "maximum_longitude"
        ),
        F.avg("longitude").alias(
            "average_longitude"
        )
    ).show(truncate=False)


# Numerical distributions

def numerical_analysis(
    df, created_col, closed_col, name
):

    print(f"\n{name}: Resolution time")

    # Include only valid timestamp pairs

    completed = (
        df.filter(
            F.col(created_col).isNotNull()
            & F.col(closed_col).isNotNull()
            & (
                F.col(closed_col)
                >= F.col(created_col)
            )
        )
        .withColumn(
            "resolution_hours",
            (
                F.unix_timestamp(
                    F.col(closed_col)
                )
                - F.unix_timestamp(
                    F.col(created_col)
                )
            ) / 3600.0
        )
    )

    completed.select(
        "resolution_hours"
    ).summary(
        "count",
        "mean",
        "stddev",
        "min",
        "25%",
        "50%",
        "75%",
        "max"
    ).show(truncate=False)


# Check malformed values

def malformed_analysis(
    df, zip_col, date_col, name
):

    print(f"\n{name}: Malformed data checks")

    zip_value = F.trim(F.col(zip_col).cast("string"))

    # Check ZIP code format

    bad_zip = df.filter(
        F.col(zip_col).isNotNull()
        & (zip_value != "")
        & (
            ~zip_value.rlike(
                r"^[0-9]{5}(-[0-9]{4})?$"
            )
        )
    )

    print(
        "Nonstandard postal-code records:",
        bad_zip.count()
    )

    bad_zip.select(
        zip_col
    ).show(10, truncate=False)

    print(
        "Missing creation timestamps:",
        df.filter(
            F.col(date_col).isNull()
        ).count()
    )

    # Check placeholder values

    for column in [
        "status",
        "complaint_type",
        "requesttype",
        "agency",
        "owner"
    ]:

        if column in df.columns:

            placeholders = df.filter(
                F.lower(
                    F.trim(F.col(column))
                ).isin(
                    "unknown",
                    "n/a",
                    "na",
                    "unspecified",
                    "null",
                    "none"
                )
            ).count()

            print(
                f"{column} placeholder values:",
                placeholders
            )


# Compare NYC and LA schemas

def schema_comparison(nyc, la):

    print("\nSchema comparison")

    nyc_columns = set(nyc.columns)
    la_columns = set(la.columns)

    print("\nIdentical column names:")
    print(sorted(nyc_columns & la_columns))

    print("\nNYC-only column names:")
    print(sorted(nyc_columns - la_columns))

    print("\nLA-only column names:")
    print(sorted(la_columns - nyc_columns))

    mappings = [
        (
            "Request ID",
            "unique_key",
            "srnumber"
        ),
        (
            "Created at",
            "created_date",
            "createddate"
        ),
        (
            "Closed at",
            "closed_date",
            "closeddate"
        ),
        (
            "Request type",
            "complaint_type",
            "requesttype"
        ),
        (
            "Status",
            "status",
            "status"
        ),
        (
            "Agency",
            "agency",
            "owner"
        ),
        (
            "Postal code",
            "incident_zip",
            "zipcode"
        ),
        (
            "Channel",
            "open_data_channel_type",
            "requestsource"
        ),
        (
            "Latitude",
            "latitude",
            "latitude"
        ),
        (
            "Longitude",
            "longitude",
            "longitude"
        )
    ]

    print("\nProposed semantic mappings:")

    print(
        f"{'Attribute':20}"
        f"{'NYC Column':28}"
        f"{'LA Column':25}"
    )

    nyc_types = dict(nyc.dtypes)
    la_types = dict(la.dtypes)

    for label, nyc_col, la_col in mappings:

        print(
            f"{label:20}"
            f"{nyc_col:28}"
            f"{la_col:25}"
        )

        print(
            f"  Types: "
            f"NYC={nyc_types[nyc_col]}, "
            f"LA={la_types[la_col]}"
        )


# Check negative resolution times in NYC

def investigate_negative_resolution(nyc_df):

    print("\nNYC negative resolution times")

    negative_resolution = (
        nyc_df.filter(
            F.col("closed_date")
            < F.col("created_date")
        )
        .withColumn(
            "resolution_hours",
            (
                F.unix_timestamp(
                    "closed_date"
                )
                - F.unix_timestamp(
                    "created_date"
                )
            ) / 3600.0
        )
    )

    print(
        "Total records with negative resolution times:",
        negative_resolution.count()
    )

    print("\nSample records:")

    negative_resolution.select(
        "unique_key",
        "created_date",
        "closed_date",
        "status",
        "resolution_hours"
    ).show(10, truncate=False)

    print("\nNegative resolution time summary:")

    negative_resolution.select(
        "resolution_hours"
    ).summary(
        "count",
        "min",
        "25%",
        "50%",
        "75%",
        "max"
    ).show(truncate=False)

    save_results(
        negative_resolution.select(
            "unique_key",
            "created_date",
            "closed_date",
            "status",
            "resolution_hours"
        ).limit(50),
        "nyc_negative_resolution_samples.csv"
    )


# Compare categorical values

def categorical_schema_comparison(nyc_df, la_df):
    """
    Compare original status and submission-channel values.
    """

    print("\nNYC submission channels")

    nyc_channels = (
        nyc_df.groupBy("open_data_channel_type")
        .count()
        .orderBy(F.desc("count"))
    )

    nyc_channels.show(100, truncate=False)

    print("\nLA submission channels")

    la_channels = (
        la_df.groupBy("requestsource")
        .count()
        .orderBy(F.desc("count"))
    )

    la_channels.show(100, truncate=False)

    print("\nNYC statuses")

    nyc_statuses = (
        nyc_df.groupBy("status")
        .count()
        .orderBy(F.desc("count"))
    )

    nyc_statuses.show(100, truncate=False)

    print("\nLA statuses")

    la_statuses = (
        la_df.groupBy("status")
        .count()
        .orderBy(F.desc("count"))
    )

    la_statuses.show(100, truncate=False)

    # Save original categorical distributions

    save_results(
        nyc_channels,
        "nyc_all_channels.csv"
    )

    save_results(
        la_channels,
        "la_all_channels.csv"
    )

    save_results(
        nyc_statuses,
        "nyc_all_statuses.csv"
    )

    save_results(
        la_statuses,
        "la_all_statuses.csv"
    )


# Run EDA analyses

def run_eda(df, name, columns):

    category_analysis(
        df,
        columns["type"],
        name,
        "request_types"
    )

    category_analysis(
        df,
        "status",
        name,
        "statuses"
    )

    category_analysis(
        df,
        columns["channel"],
        name,
        "channels"
    )

    category_analysis(
        df,
        columns["agency"],
        name,
        "agencies"
    )

    monthly_analysis(
        df,
        columns["created"],
        name
    )

    geographic_analysis(
        df,
        columns["zip"],
        name
    )

    numerical_analysis(
        df,
        columns["created"],
        columns["closed"],
        name
    )

    malformed_analysis(
        df,
        columns["zip"],
        columns["created"],
        name
    )


# Analyze cleaned datasets

def processed_analysis(nyc_df, la_df):
    """
    Save distributions from the cleaned NYC and LA datasets.
    """

    print("\nCleaned dataset analysis")

    processed_dir = OUTPUT_DIR / "processed"
    processed_dir.mkdir(exist_ok=True)

    for city, df in [
        ("nyc", nyc_df),
        ("la", la_df)
    ]:

        for column, filename in [
            (
                "submission_channel_standardized",
                "channels"
            ),
            (
                "status_standardized",
                "statuses"
            ),
            (
                "postal_code",
                "postal_codes"
            )
        ]:

            result = (
                df.groupBy(column)
                .count()
                .orderBy(F.desc("count"))
            )

            rows = result.collect()

            output_path = (
                processed_dir / f"{city}_{filename}.csv"
            )

            with open(
                output_path,
                "w",
                newline="",
                encoding="utf-8"
            ) as file:

                writer = csv.writer(file)

                writer.writerow([
                    column,
                    "count"
                ])

                for row in rows:

                    writer.writerow([
                        row[column],
                        row["count"]
                    ])

            print("Saved:", output_path)


# Run analyses

try:

    # Load original datasets

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

    # NYC distribution analysis

    print("\nNYC analysis")

    run_eda(
        nyc_df,
        "NYC",
        {
            "type": "complaint_type",
            "channel": "open_data_channel_type",
            "agency": "agency",
            "created": "created_date",
            "closed": "closed_date",
            "zip": "incident_zip"
        }
    )

    # LA distribution analysis

    print("\nLA analysis")

    run_eda(
        la_df,
        "LA",
        {
            "type": "requesttype",
            "channel": "requestsource",
            "agency": "owner",
            "created": "createddate",
            "closed": "closeddate",
            "zip": "zipcode"
        }
    )

    # Compare source schemas

    schema_comparison(
        nyc_df,
        la_df
    )

    # Check negative resolution times

    investigate_negative_resolution(
        nyc_df
    )

    # Compare original categories

    categorical_schema_comparison(
        nyc_df,
        la_df
    )

    # Load cleaned datasets

    nyc_cleaned = spark.read.parquet(
        str(BASE_DIR / "processed_data" / "nyc_cleaned")
    )

    la_cleaned = spark.read.parquet(
        str(BASE_DIR / "processed_data" / "la_cleaned")
    )

    # Save cleaned data distributions

    processed_analysis(
        nyc_cleaned,
        la_cleaned
    )

    print("\nEDA analysis completed.")

finally:

    spark.stop()
