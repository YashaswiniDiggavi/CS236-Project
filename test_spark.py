from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("CS236_Phase1_Test") \
    .master("local[*]") \
    .getOrCreate()

data = [
    ("NYC", "Pothole"),
    ("LA", "Street Light"),
    ("NYC", "Garbage")
]

df = spark.createDataFrame(data, ["City", "Request_Type"])

df.show()

spark.stop()