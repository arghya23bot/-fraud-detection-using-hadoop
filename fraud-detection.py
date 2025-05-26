from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum, avg, stddev, when
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("FraudDetectionUsingHadoop") \
    .getOrCreate()

# Define schema for the dataset
schema = StructType([
    StructField("transaction_id", StringType(), True),
    StructField("user_id", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("timestamp", TimestampType(), True)
])

# Load data from HDFS or local filesystem
df = spark.read.csv("hdfs://localhost:9000/user/hadoop/transactions.csv", header=True, schema=schema)

# Basic statistics per user
user_stats = df.groupBy("user_id").agg(
    avg("amount").alias("avg_amount"),
    stddev("amount").alias("stddev_amount"),
    count("*").alias("transaction_count")
)

# Join stats with original data
df_with_stats = df.join(user_stats, on="user_id")

# Flag transactions with unusually high amounts (more than mean + 3*stddev)
df_fraud = df_with_stats.withColumn(
    "is_fraud",
    when(col("amount") > col("avg_amount") + 3 * col("stddev_amount"), 1).otherwise(0)
)

# Show suspicious transactions
df_fraud.filter(col("is_fraud") == 1).show()

# Optionally: Save fraud data to HDFS
df_fraud.filter(col("is_fraud") == 1) \
    .write.csv("hdfs://localhost:9000/user/hadoop/fraud_transactions", header=True, mode="overwrite")

# Stop Spark session
spark.stop()
