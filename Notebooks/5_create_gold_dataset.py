# Databricks notebook source
# Check current schema
silver_raw_info = spark.table("churn_silver.raw_info")
print("Current columns:")
print(silver_raw_info.columns)
print(f"\nCurrent count: {silver_raw_info.count()}")

# Check schema
silver_raw_info.printSchema()

# COMMAND ----------

from pyspark.sql.functions import row_number
from pyspark.sql.window import Window

# Read the current raw_info table
silver_raw_info = spark.table("churn_silver.raw_info")

print(f"Before dedup: {silver_raw_info.count():,}")
print(f"Unique customers before: {silver_raw_info.select('unique_customer_identifier').distinct().count():,}")

# Get original columns
original_cols = silver_raw_info.columns

# Deduplicate - keep first row per customer
window_spec = Window.partitionBy("unique_customer_identifier").orderBy("behav_win")
silver_raw_info_dedup = silver_raw_info.withColumn("rn", row_number().over(window_spec)) \
    .filter(col("rn") == 1) \
    .select(original_cols)  # Select only original columns

print(f"\nAfter dedup: {silver_raw_info_dedup.count():,}")
print(f"Unique customers after: {silver_raw_info_dedup.select('unique_customer_identifier').distinct().count():,}")

# Overwrite with same schema
silver_raw_info_dedup.write.format("delta").mode("overwrite").saveAsTable("churn_silver.raw_info")
print("\nOverwritten churn_silver.raw_info")

# COMMAND ----------

import sys
import importlib
import sys
sys.path.append("/Workspace/Users/michaelknight@leviathan.onmicrosoft.com/telecoms-churn")

# Reload the module to pick up changes
import src.feature_func as ff
importlib.reload(ff)

from src.feature_func import (
    create_contract_status_features,
    create_ooc_days_features,
    create_technology_features,
    create_speed_features,
    create_sales_channel_features,
    create_crm_package_features,
    create_usage_days_feature,
    create_has_calls_feature
)

# Start with base: customer ID, target, and continuous variables
silver_raw_info = spark.table("churn_silver.raw_info")
base = silver_raw_info.select(
    "unique_customer_identifier",
    "churned",
    "dd_cancel_60_day",
    "contract_dd_cancels",
    "line_speed",
    "tenure_days"
)

print(f"Base records: {base.count()}")

# Create each feature set
print("Creating contract_status features...")
contract_features = create_contract_status_features(spark)

print("Creating ooc_days features...")
ooc_features = create_ooc_days_features(spark)

print("Creating technology features...")
technology_features = create_technology_features(spark)

print("Creating speed features...")
speed_features = create_speed_features(spark)

print("Creating sales_channel features...")
sales_channel_features = create_sales_channel_features(spark)

print("Creating crm_package features...")
crm_package_features = create_crm_package_features(spark)

print("Creating usage_days features...")
usage_features = create_usage_days_feature(spark)

print("Creating has_calls features...")
calls_features = create_has_calls_feature(spark)

# Join all features together
print("Joining all features...")
gold = base \
    .join(contract_features, on="unique_customer_identifier", how="left") \
    .join(ooc_features, on="unique_customer_identifier", how="left") \
    .join(technology_features, on="unique_customer_identifier", how="left") \
    .join(speed_features, on="unique_customer_identifier", how="left") \
    .join(sales_channel_features, on="unique_customer_identifier", how="left") \
    .join(crm_package_features, on="unique_customer_identifier", how="left") \
    .join(usage_features, on="unique_customer_identifier", how="left") \
    .join(calls_features, on="unique_customer_identifier", how="left")

print(f"\nGold dataset created: {gold.count()} rows, {len(gold.columns)} columns")



# COMMAND ----------

# Drop the corrupted gold table
spark.sql("DROP TABLE IF EXISTS churn_gold.modeling_dataset")

# Then save your gold dataset
gold.write.format("delta").mode("overwrite").saveAsTable("churn_gold.modeling_dataset")
print("Saved to churn_gold.modeling_dataset")

# COMMAND ----------



# Read the gold table
gold_dataset = spark.table("churn_gold.modeling_dataset")

# Show 10 random customers
gold_sample = gold_dataset.orderBy(rand()).limit(10).toPandas()

print(f"Total records: {gold_dataset.count()}")
print(f"Total columns: {len(gold_dataset.columns)}")
print("\n10 Random Customers:")
gold_sample

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read gold table and convert to pandas
gold_df = spark.table("churn_gold.modeling_dataset")
gold_pd = gold_df.toPandas()

# Calculate correlation matrix
corr_matrix = gold_pd.corr()

# Create figure
plt.figure(figsize=(20, 18))

# Create heatmap
sns.heatmap(
    corr_matrix,
    cmap='RdBu_r',
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    annot=False  # Set to True if you want to see values, but with 49 cols it will be crowded
)

plt.title('Correlation Matrix - All Features', fontsize=16, pad=20)
plt.tight_layout()
plt.show()

# Optional: Show correlations with target variable (churned)
print("\n=== Top Correlations with Churned ===")
churn_corr = corr_matrix['churned'].sort_values(ascending=False)
print(churn_corr)

# COMMAND ----------

# Check which features have perfect correlation
corr_matrix = gold_pd.corr()
perfect_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.99:
            perfect_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

print("Perfect correlations (>0.99):")
for col1, col2, corr_val in perfect_corr:
    print(f"{col1} <-> {col2}: {corr_val:.3f}")

# COMMAND ----------

columns_to_drop = [
    'package_speed_18',   # Redundant with technology_mpf
    'package_speed_65',   # Redundant with crm_package_fibre_65
    'package_speed_40',   # Redundant with crm_package_faster_fibre
    'package_speed_35',   # Redundant with crm_package_fibre_35
    'package_speed_1000', # Redundant with crm_package_ultra_fibre_optic
    'package_speed_150'   # Redundant with crm_package_fibre_150
]

# Drop from gold table
gold_reduced = gold.drop(*columns_to_drop)

print(f"Columns before: {len(gold.columns)}")
print(f"Columns after: {len(gold_reduced.columns)}")

# Overwrite gold table
spark.sql("DROP TABLE IF EXISTS churn_gold.modeling_dataset")
gold_reduced.write.format("delta").mode("overwrite").saveAsTable("churn_gold.modeling_dataset")
print("Saved reduced gold dataset")

# COMMAND ----------

import pandas as pd

gold_pd = spark.table("churn_gold.modeling_dataset").toPandas()

# Define each one-hot encoded group
ohe_groups = {
    'contract_status': [c for c in gold_pd.columns if c.startswith('contract_status_')],
    'technology': [c for c in gold_pd.columns if c.startswith('technology_')],
    'package_speed': [c for c in gold_pd.columns if c.startswith('package_speed_')],
    'sales_channel': [c for c in gold_pd.columns if c.startswith('sales_channel_')],
    'crm_package': [c for c in gold_pd.columns if c.startswith('crm_package_')],
    'ooc_bucket': [c for c in gold_pd.columns if c.startswith('ooc_bucket_')],
    'usage_days_bucket': [c for c in gold_pd.columns if c.startswith('usage_days_bucket_')]
}

# For each group, check that each row sums to exactly 1
print("=== ONE-HOT ENCODING VALIDATION ===")
for group_name, cols in ohe_groups.items():
    row_sums = gold_pd[cols].sum(axis=1)
    all_ones = (row_sums == 1).all()
    min_sum = row_sums.min()
    max_sum = row_sums.max()
    print(f"\n{group_name}:")
    print(f"  Columns: {cols}")
    print(f"  All rows sum to 1: {all_ones}")
    print(f"  Min row sum: {min_sum}, Max row sum: {max_sum}")

# COMMAND ----------


