# Databricks notebook source
dbutils.library.restartPython()

# COMMAND ----------

import sys
sys.path.append('/Workspace/Users/michaelknight@leviathan.onmicrosoft.com/telecoms-churn')
from pyspark.sql.functions import (
    lit, col, rand, row_number, add_months, max as spark_max, trunc
)
from src.dataset_creation import create_live_customer_baseline
from src.dataset_creation import create_churned_customer_baseline
from pyspark.sql.functions import min as spark_min, trunc, row_number
from pyspark.sql.window import Window


# Load data
customer_info_df = spark.table("leviathan_wkspc.churn_bronze.customer_info_bronze")
cease_df = spark.table("leviathan_wkspc.churn_bronze.cease_bronze")

# Create live customer baseline
silver_live_raw_info = create_live_customer_baseline(customer_info_df, cease_df, buffer_months=3)
silver_live_raw_info = silver_live_raw_info.drop("datevalue")

# Validate
print(f"Live customers: {silver_live_raw_info.count():,}")
silver_live_raw_info.show(5)

# COMMAND ----------

# Create churned customer baseline (now includes reason_description_insight)
silver_churned_raw_info = create_churned_customer_baseline(customer_info_df, cease_df)
print(f"Churned customers: {silver_churned_raw_info.count():,}")

# Step 1: Remove first month churners
min_date = customer_info_df.select(spark_min("datevalue")).collect()[0][0]
silver_churned_raw_info = silver_churned_raw_info.filter(col("behav_win") != min_date)
print(f"After removing first month: {silver_churned_raw_info.count():,}")

# Step 2: Filter out specific churn reasons
reasons_to_exclude = ['Bereavement', 'HomeMove', 'BadDebtDisconnect']
silver_churned_raw_info = silver_churned_raw_info.filter(
    ~col("reason_description_insight").isin(reasons_to_exclude)
)
print(f"After filtering churn reasons: {silver_churned_raw_info.count():,}")

# Step 3: Deduplicate (in case there are still any duplicates)
window_spec = Window.partitionBy("unique_customer_identifier").orderBy("behav_win")
silver_churned_raw_info = silver_churned_raw_info.withColumn("rn", row_number().over(window_spec)) \
    .filter(col("rn") == 1).drop("rn")
print(f"After dedup: {silver_churned_raw_info.count():,}")

# Step 4: Drop reason column
silver_churned_raw_info = silver_churned_raw_info.drop("reason_description_insight")

# Reorder and union
col_order = ['unique_customer_identifier', 'behav_win', 'churned', 'contract_status', 
             'contract_dd_cancels', 'dd_cancel_60_day', 'ooc_days', 'technology', 
             'speed', 'line_speed', 'sales_channel', 'crm_package_name', 'tenure_days']

silver_live_raw_info = silver_live_raw_info.select(col_order)
silver_churned_raw_info = silver_churned_raw_info.select(col_order)

silver_raw_info = silver_live_raw_info.union(silver_churned_raw_info)
print(f"\nTotal rows after union: {silver_raw_info.count():,}")
silver_raw_info.groupBy("churned").count().show()

# COMMAND ----------

from pyspark.sql.functions import count, when, col, isnan

# Check missing values for each column
print("Missing values by column:")

missing_counts = []
for c in silver_raw_info.columns:
    col_type = dict(silver_raw_info.dtypes)[c]
    if col_type in ['double', 'float']:
        missing_counts.append(
            count(when(col(c).isNull() | isnan(c), c)).alias(c)
        )
    else:
        missing_counts.append(
            count(when(col(c).isNull(), c)).alias(c)
        )

silver_raw_info.select(missing_counts).show(vertical=True)

# Also show total rows for context
print(f"\nTotal rows: {silver_raw_info.count():,}")

# COMMAND ----------

from pyspark.sql.functions import rand
silver_raw_info.orderBy(rand()).limit(20).toPandas()

# COMMAND ----------

# Save to Delta table
spark.sql("DROP TABLE IF EXISTS churn_silver.silver_raw_info")
spark.sql("CREATE DATABASE IF NOT EXISTS churn_silver")

silver_raw_info.write.format("delta").mode("overwrite").saveAsTable("churn_silver.raw_info")

print("✓ silver_raw_info saved to Delta table")
print(f"Total rows: {silver_raw_info.count():,}")
silver_raw_info.groupBy("churned").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Usage data

# COMMAND ----------

# MAGIC %md
# MAGIC A very large file we need to edit to reduce to our behav_win

# COMMAND ----------

from pyspark.sql.functions import year, month

# Join usage data with silver table to filter by behav_win month
usage_df = spark.table("churn_bronze.usage_bronze")

# Join and filter to same month as behav_win
filtered_usage = usage_df.join(
    silver_raw_info.select("unique_customer_identifier", "behav_win"),
    on="unique_customer_identifier",
    how="inner"
).filter(
    (year(usage_df.calendar_date) == year(silver_raw_info.behav_win)) &
    (month(usage_df.calendar_date) == month(silver_raw_info.behav_win))
)

print(f"Total usage records: {usage_df.count()}")
print(f"Usage records matching behav_win month: {filtered_usage.count()}")

# Save as a reduced table for easier processing
filtered_usage.write.format("delta").mode("overwrite").saveAsTable("churn_silver.usage_behav_month")
print("Saved to churn_silver.usage_behav_month")
