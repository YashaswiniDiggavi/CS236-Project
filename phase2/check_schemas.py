
from pathlib import Path
from pyspark.sql import SparkSession

PROJECT_ROOT = Path(__file__).resolve().parent.parent

spark = (
    SparkSession.builder
    .appName("CS236_Phase2_Check_Schemas")
    .master("local[4]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

datasets = {
    "NYC Cleaned": "nyc_cleaned",
    "LA Cleaned": "la_cleaned",
    "Unified": "merged_311",
}

try:
    for name, folder in datasets.items():
        path = str(PROJECT_ROOT / "processed_data" / folder)
        df = spark.read.parquet(path)

        print(f"\n========== {name} ==========")
        print("Record count:", df.count())
        print("Number of columns:", len(df.columns))
        df.printSchema()

finally:
    spark.stop()
