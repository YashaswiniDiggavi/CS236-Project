
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

# CS236 - Phase 1: Data Preparation and EDA in Spark
# Exploratory data analysis

# Spark setup

spark = (
    SparkSession.builder
    .appName("CS236_Phase1_EDA")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# Load datasets

nyc_df = spark.read.csv(
    "data/nyc_311_800k.csv",
    header=True,
    inferSchema=True
)

la_df = spark.read.csv(
    "data/la_311_800k.csv",
    header=True,
    inferSchema=True
)

# Explore dataset structure

def load_and_scan(df, dataset_name, created_column):

    print(f"\n========== {dataset_name} DATASET ==========")

    # Count rows
    print("Total Rows:", df.count())

    # Count columns
    print("Total Columns:", len(df.columns))

    # Display column names and data types
    print(f"\n{dataset_name} Schema:")
    df.printSchema()

    # Display first five records
    print(f"\n{dataset_name} Sample Records:")
    df.show(5, truncate=False)

    # Find minimum and maximum creation timestamps
    print(f"\n{dataset_name} Creation Date Range:")

    df.select(
        F.min(created_column).alias("earliest_date"),
        F.max(created_column).alias("latest_date")
    ).show(truncate=False)

# Check missing values

def check_missing_values(df, dataset_name):

    print(f"\n========== {dataset_name}: MISSING VALUES ==========")

    expressions = []

    for field in df.schema.fields:

        column = F.col(field.name)

    # Missing values for string columns include NULL,
    # empty strings, and strings containing only spaces.
        if isinstance(field.dataType, StringType):
            missing_condition = (
                column.isNull() |
                (F.trim(column) == "")
            )

        else:
            missing_condition = column.isNull()

        expressions.append(
            F.sum(
                F.when(missing_condition, 1).otherwise(0)
            ).alias(field.name)
        )

    # Calculate missing counts for all columns
    results = df.agg(*expressions).collect()[0]

    total_rows = df.count()

    print(f"Total Records: {total_rows}")

    print(
        f"{'Column Name':35} "
        f"{'Missing Count':15} "
        f"{'Missing %':12}"
    )

    print("-" * 65)

    for column in df.columns:

        missing_count = results[column] or 0

        percentage = (
            missing_count / total_rows * 100
            if total_rows > 0 else 0
        )

        print(
            f"{column:35} "
            f"{missing_count:<15} "
            f"{percentage:.2f}%"
        )

# Check duplicates

def check_duplicates(df, dataset_name):

    print(f"\n========== {dataset_name}: DUPLICATES ==========")

    total_rows = df.count()

    distinct_rows = df.distinct().count()

    duplicate_rows = total_rows - distinct_rows

    print("Total Records:", total_rows)
    print("Distinct Records:", distinct_rows)
    print("Duplicate Records:", duplicate_rows)


# Check unique request IDs

def check_identifiers(df, id_column, dataset_name):

    print(
        f"\n========== {dataset_name}: IDENTIFIER CHECK =========="
    )

# Exclude missing identifiers from the uniqueness check
    valid_ids = df.filter(
        F.col(id_column).isNotNull() &
        (
            F.trim(F.col(id_column).cast("string")) != ""
        )
    )

    total_rows = df.count()

    valid_id_count = valid_ids.count()

    missing_ids = total_rows - valid_id_count

    distinct_ids = (
        valid_ids.select(id_column).distinct().count()
    )

    duplicate_id_groups = (
        valid_ids.groupBy(id_column)
        .count()
        .filter(F.col("count") > 1)
        .count()
    )

    excess_duplicate_id_rows = valid_id_count - distinct_ids

    print("Total Records:", total_rows)
    print("Missing Identifiers:", missing_ids)
    print("Valid Identifier Records:", valid_id_count)
    print("Distinct Valid Identifiers:", distinct_ids)
    print("Identifiers Appearing More Than Once:",
          duplicate_id_groups)
    print("Excess Rows Sharing Identifiers:",
          excess_duplicate_id_rows)

# Check invalid values

def check_invalid_values(
    df, created_column, closed_column, dataset_name
):

    print(
        f"\n========== {dataset_name}: INVALID VALUES =========="
    )

# Check missing creation timestamps
    missing_created_dates = df.filter(
        F.col(created_column).isNull()
    ).count()

# Check closed dates before creation dates
    invalid_date_order = df.filter(
        F.col(closed_column) < F.col(created_column)
    ).count()

# Check latitude outside the global valid range
    invalid_latitude = df.filter(
        F.col("latitude").isNotNull() &
        (
            (F.col("latitude") < -90) |
            (F.col("latitude") > 90)
        )
    ).count()

# Check longitude outside the global valid range
    invalid_longitude = df.filter(
        F.col("longitude").isNotNull() &
        (
            (F.col("longitude") < -180) |
            (F.col("longitude") > 180)
        )
    ).count()

# Check whether only one coordinate is missing
    incomplete_coordinates = df.filter(
        (
            F.col("latitude").isNull() &
            F.col("longitude").isNotNull()
        ) |
        (
            F.col("latitude").isNotNull() &
            F.col("longitude").isNull()
        )
    ).count()

    print("Missing Creation Dates:", missing_created_dates)
    print("Closure Before Creation:", invalid_date_order)
    print("Invalid Latitude Values:", invalid_latitude)
    print("Invalid Longitude Values:", invalid_longitude)
    print("Incomplete Coordinate Pairs:",
          incomplete_coordinates)


# Run EDA checks

try:

# Check dataset structure
    load_and_scan(
        nyc_df, "NYC", "created_date"
    )

    load_and_scan(
        la_df, "LA", "createddate"
    )

# Check missing values
    check_missing_values(nyc_df, "NYC")
    check_missing_values(la_df, "LA")

# Check duplicates
    check_duplicates(nyc_df, "NYC")
    check_duplicates(la_df, "LA")

# Check unique request IDs
    check_identifiers(
        nyc_df, "unique_key", "NYC"
    )

    check_identifiers(
        la_df, "srnumber", "LA"
    )

# Check invalid values
    check_invalid_values(
        nyc_df,
        "created_date",
        "closed_date",
        "NYC"
    )

    check_invalid_values(
        la_df,
        "createddate",
        "closeddate",
        "LA"
    )

finally:

# Stop Spark even if an analysis raises an error
    spark.stop()
