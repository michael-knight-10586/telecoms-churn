# Databricks notebook source
# Create a database to organize our bronze layer tables
# This separates our project from other work in the workspace
spark.sql("CREATE DATABASE IF NOT EXISTS churn_bronze")
spark.sql("USE churn_bronze")

print("Database 'churn_bronze' created and selected")

# COMMAND ----------

# Read cease.csv from DBFS using PySpark
cease_df = spark.read.csv(
    "dbfs:/FileStore/tables/cease.csv",
    header=True,           # First row contains column names
    inferSchema=True       # Automatically detect data types
)

# Display first few rows to verify
display(cease_df.limit(5))

# Write as Delta table in bronze layer
cease_df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("cease_bronze")

print(f"Cease data ingested: {cease_df.count()} rows")

# COMMAND ----------

# Read calls.csv from DBFS
calls_df = spark.read.csv(
    "dbfs:/FileStore/tables/calls.csv",
    header=True,
    inferSchema=True
)

# Display sample
display(calls_df.limit(5))

# Write as Delta table
calls_df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("calls_bronze")

print(f"Calls data ingested: {calls_df.count()} rows")

# COMMAND ----------

# Read customer_info.parquet from DBFS
# Parquet files don't need header or inferSchema - metadata is built in
customer_info_df = spark.read.parquet(
    "dbfs:/FileStore/tables/customer_info.parquet"
)

# Display sample
display(customer_info_df.limit(25))

# Write as Delta table
customer_info_df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("customer_info_bronze")

print(f"Customer info data ingested: {customer_info_df.count()} rows")

# COMMAND ----------

file_info = dbutils.fs.ls("dbfs:/FileStore/tables/")
for f in file_info:
    if 'usage' in f.name:
        print(f"File: {f.name}, Size: {f.size} bytes")

# COMMAND ----------

# Read usage.parquet from DBFS
usage_df = spark.read.parquet("dbfs:/FileStore/tables/usage_new.parquet")

# Display sample
display(usage_df.limit(5))

# Write as Delta table
usage_df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("usage_bronze")

print(f"Usage data ingested: {usage_df.count()} rows")

# COMMAND ----------


