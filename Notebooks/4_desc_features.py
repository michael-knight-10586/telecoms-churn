# Databricks notebook source
# ============================================================================
# SETUP - Imports and Data Loading
# ============================================================================

# System path for imports
import sys
sys.path.append(sys.path.append('/Workspace/Users/michael.knight.10586@outlook.com/telecoms-churn')

# PySpark functions
from pyspark.sql.functions import (
    lit, col, rand, row_number, add_months, 
    max as spark_max, min as spark_min,
    avg, count as spark_count, countDistinct,
    when, isnan, isnull,
    year, month, dayofmonth, last_day,
    sum as spark_sum, trunc
)
from pyspark.sql.window import Window

# Python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Pandas display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

# ============================================================================
# Load Tables from Bronze and Silver Layers
# ============================================================================

print("Loading tables from medallion architecture...")

# Silver layer - main modeling dataset
silver_raw_info = spark.table("churn_silver.raw_info")
usage_behav_month = spark.table("churn_silver.usage_behav_month")

# Bronze layer - source data
customer_info_df = spark.table("churn_bronze.customer_info_bronze")
cease_df = spark.table("churn_bronze.cease_bronze")
calls_df = spark.table("churn_bronze.calls_bronze")
usage_df = spark.table("churn_bronze.usage_bronze")

# Display table counts
print(f"\n✓ silver_raw_info: {silver_raw_info.count():,} rows")
print(f"✓ usage_behav_month: {usage_behav_month.count():,} rows")
print(f"✓ customer_info_bronze: {customer_info_df.count():,} rows")
print(f"✓ cease_bronze: {cease_df.count():,} rows")
print(f"✓ calls_bronze: {calls_df.count():,} rows")
print(f"✓ usage_bronze: {usage_df.count():,} rows")

print("\nSetup complete. Ready for feature exploration.")

# COMMAND ----------

# MAGIC %md
# MAGIC ##1) Contract status

# COMMAND ----------

churn_by_status = silver_raw_info.groupBy("contract_status").agg(
    count("*").alias("total_customers"),
    avg("churned").alias("churn_rate")
)

total = silver_raw_info.count()

churn_by_status = churn_by_status.withColumn(
    "pct_of_total", col("total_customers") / total * 100
).orderBy(col("churn_rate").desc())

churn_by_status.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ##2 & 3) DD cancels

# COMMAND ----------

total = silver_raw_info.count()

dd_cancels = silver_raw_info.groupBy("contract_dd_cancels").agg(
    count("*").alias("total_customers"),
    avg("churned").alias("churn_rate")
).withColumn("pct_of_total", col("total_customers") / total * 100) \
 .orderBy("contract_dd_cancels")

dd_cancel_60 = silver_raw_info.groupBy("dd_cancel_60_day").agg(
    count("*").alias("total_customers"),
    avg("churned").alias("churn_rate")
).withColumn("pct_of_total", col("total_customers") / total * 100) \
 .orderBy("dd_cancel_60_day")

dd_cancels.toPandas()

# COMMAND ----------

dd_cancel_60.toPandas()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ##4) OOC Days

# COMMAND ----------

# Prepare aggregated data
ooc_aggregated = silver_raw_info.filter(
    (col("ooc_days").isNotNull()) & 
    (col("ooc_days").between(-150, 170))
) \
    .groupBy("ooc_days").agg(avg("churned").alias("churn_rate")) \
    .toPandas().sort_values('ooc_days')

X = ooc_aggregated[['ooc_days']].values
y = ooc_aggregated['churn_rate'].values

# Prepare data for polynomial fit
ooc_aggregated = silver_raw_info.filter(
    (col("ooc_days").isNotNull()) & 
    (col("ooc_days").between(-150, 170))
).groupBy("ooc_days").agg(avg("churned").alias("churn_rate")).toPandas().sort_values('ooc_days')

X = ooc_aggregated[['ooc_days']].values
y = ooc_aggregated['churn_rate'].values

# Get distribution data
ooc_distribution = silver_raw_info.filter(
    (col("ooc_days").isNotNull()) & 
    (col("ooc_days").between(-150, 170))
).groupBy("ooc_days").agg(spark_count("*").alias("customer_count")).toPandas().sort_values('ooc_days')

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

# Left plot: Polynomial relationship
degree = 7
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)

X_smooth = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
X_smooth_poly = poly.transform(X_smooth)
y_pred = model.predict(X_smooth_poly)

ax1.scatter(X, y, alpha=0.6, s=20, color='blue', label='Actual')
ax1.plot(X_smooth, y_pred, 'r-', linewidth=2, label=f'Polynomial degree {degree}')
ax1.axhline(y=0.63, color='green', linestyle='--', alpha=0.5, label='Overall churn')
ax1.set_xlabel('OOC Days')
ax1.set_ylabel('Churn Rate')
ax1.set_title(f'Relationship between OOC Days and Churn (Degree {degree})')
ax1.grid(True, alpha=0.3)
ax1.legend()

r2 = r2_score(y, model.predict(X_poly))
ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))

# Right plot: Customer distribution
ax2.plot(ooc_distribution['ooc_days'], ooc_distribution['customer_count'], linewidth=1, color='steelblue')
ax2.set_xlabel('OOC Days')
ax2.set_ylabel('Number of Customers')
ax2.set_title('Distribution of Customers by OOC Days')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Stats
print(f"Total data points: {len(ooc_distribution)}")
print(f"Min customers in bucket: {ooc_distribution['customer_count'].min()}")
print(f"Max customers in bucket: {ooc_distribution['customer_count'].max()}")

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

ooc_daily = silver_raw_info.filter(
    col("ooc_days").isNotNull() & col("ooc_days").between(-170, -80)
).groupBy("ooc_days").agg(
    avg("churned").alias("churn_rate"),
    count("*").alias("n")
).toPandas().sort_values("ooc_days")

buckets = []
bucket_n = 0
bucket_churned_sum = 0
bucket_days = []

for _, row in ooc_daily.iterrows():
    bucket_days.append(row["ooc_days"])
    bucket_n += row["n"]
    bucket_churned_sum += row["churn_rate"] * row["n"]
    
    if bucket_n >= 1000:
        buckets.append({
            "ooc_mid": np.mean(bucket_days),
            "churn_rate": bucket_churned_sum / bucket_n,
            "n": bucket_n
        })
        bucket_n = 0
        bucket_churned_sum = 0
        bucket_days = []

if bucket_n > 0:
    buckets.append({
        "ooc_mid": np.mean(bucket_days),
        "churn_rate": bucket_churned_sum / bucket_n,
        "n": bucket_n
    })

import pandas as pd
ooc_bucketed = pd.DataFrame(buckets)

X = ooc_bucketed[["ooc_mid"]].values
y = ooc_bucketed["churn_rate"].values
w = ooc_bucketed["n"].values

degree = 6
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y, sample_weight=w)

X_smooth = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
y_pred = model.predict(poly.transform(X_smooth))

overall_churn_rate = silver_raw_info.agg(avg("churned")).collect()[0][0]
plt.figure(figsize=(12, 5))
plt.scatter(X, y, s=20, alpha=0.7, color="steelblue", label="Actual (~1000 customers per dot)")
# Split curve into three segments: before, highlight, after
elbow_low, elbow_high = -123, -110
mask_before = X_smooth.flatten() < elbow_low
mask_elbow  = (X_smooth.flatten() >= elbow_low) & (X_smooth.flatten() <= elbow_high)
mask_after  = X_smooth.flatten() > elbow_high

plt.plot(X_smooth[mask_before], y_pred[mask_before], "r-", linewidth=2, label=f"Degree {degree} weighted poly fit")
plt.plot(X_smooth[mask_elbow],  y_pred[mask_elbow],  color="royalblue", linewidth=3)
plt.plot(X_smooth[mask_after],  y_pred[mask_after],  "r-", linewidth=2)

# Arrow and label
elbow_x = -117
elbow_y = model.predict(poly.transform([[elbow_x]]))[0]
plt.annotate(
    "2 week intervention window",
    xy=(elbow_x, elbow_y),
    xytext=(elbow_x + 15, elbow_y - 0.05),
    fontsize=10,
    arrowprops=dict(arrowstyle="->", color="black"),
    color="black"
)

# Second arrow - other side at -110
attention_x = -110
attention_y = model.predict(poly.transform([[attention_x]]))[0]
plt.annotate(
    "attention elbow",
    xy=(attention_x, attention_y),
    xytext=(attention_x - 30, attention_y + 0.05),
    fontsize=10,
    arrowprops=dict(arrowstyle="->", color="black"),
    color="black"
)

plt.axhline(y=overall_churn_rate, color="green", linestyle="--", alpha=0.6, label="Overall churn rate")
plt.xlabel("OOC Days")
plt.ylabel("Churn Rate")
plt.title("Churn Rate by OOC Days (-200 to -50)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Buckets created: {len(ooc_bucketed)}")
print(f"R² = {r2_score(y, model.predict(X_poly), sample_weight=w):.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##5) Technology

# COMMAND ----------

total = silver_raw_info.count()

technology = silver_raw_info.groupBy("technology").agg(
    count("*").alias("total_customers"),
    avg("churned").alias("churn_rate")
).withColumn("pct_of_total", col("total_customers") / total * 100) \
 .orderBy(col("total_customers").desc())

technology.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6) Speed

# COMMAND ----------

total = silver_raw_info.count()

speed = silver_raw_info.groupBy("speed").agg(
    count("*").alias("total_customers"),
    avg("churned").alias("churn_rate")
).withColumn("pct_of_total", col("total_customers") / total * 100) \
 .orderBy(col("total_customers").desc())

speed.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ##7) Line Speed

# COMMAND ----------

total = silver_raw_info.count()
quantiles = silver_raw_info.approxQuantile("line_speed", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 0.01)

line_speed_buckets = silver_raw_info.withColumn(
    "line_speed_bucket",
    when(col("line_speed").isNull(), "Missing")
    .when(col("line_speed") <= quantiles[0], f"0-{quantiles[0]:.1f}")
    .when(col("line_speed") <= quantiles[1], f"{quantiles[0]:.1f}-{quantiles[1]:.1f}")
    .when(col("line_speed") <= quantiles[2], f"{quantiles[1]:.1f}-{quantiles[2]:.1f}")
    .when(col("line_speed") <= quantiles[3], f"{quantiles[2]:.1f}-{quantiles[3]:.1f}")
    .when(col("line_speed") <= quantiles[4], f"{quantiles[3]:.1f}-{quantiles[4]:.1f}")
    .when(col("line_speed") <= quantiles[5], f"{quantiles[4]:.1f}-{quantiles[5]:.1f}")
    .when(col("line_speed") <= quantiles[6], f"{quantiles[5]:.1f}-{quantiles[6]:.1f}")
    .when(col("line_speed") <= quantiles[7], f"{quantiles[6]:.1f}-{quantiles[7]:.1f}")
    .when(col("line_speed") <= quantiles[8], f"{quantiles[7]:.1f}-{quantiles[8]:.1f}")
    .otherwise(f"{quantiles[8]:.1f}+")
).groupBy("line_speed_bucket").agg(
    count("*").alias("total_customers"),
    avg("churned").alias("churn_rate")
).withColumn("pct_of_total", col("total_customers") / total * 100).toPandas()

bucket_order = [
    f"0-{quantiles[0]:.1f}",
    f"{quantiles[0]:.1f}-{quantiles[1]:.1f}",
    f"{quantiles[1]:.1f}-{quantiles[2]:.1f}",
    f"{quantiles[2]:.1f}-{quantiles[3]:.1f}",
    f"{quantiles[3]:.1f}-{quantiles[4]:.1f}",
    f"{quantiles[4]:.1f}-{quantiles[5]:.1f}",
    f"{quantiles[5]:.1f}-{quantiles[6]:.1f}",
    f"{quantiles[6]:.1f}-{quantiles[7]:.1f}",
    f"{quantiles[7]:.1f}-{quantiles[8]:.1f}",
    f"{quantiles[8]:.1f}+",
    "Missing"
]

line_speed_buckets['line_speed_bucket'] = pd.Categorical(
    line_speed_buckets['line_speed_bucket'], categories=bucket_order, ordered=True
)
line_speed_buckets = line_speed_buckets.sort_values('line_speed_bucket')

line_speed_buckets

# COMMAND ----------

total = silver_raw_info.count()

line_speed_buckets = silver_raw_info.withColumn(
    "line_speed_bucket",
    when(col("line_speed").isNull(), "Missing")
    .when(col("line_speed").between(0, 9), "0-9")
    .when(col("line_speed").between(10, 19), "10-19")
    .when(col("line_speed").between(20, 29), "20-29")
    .when(col("line_speed").between(30, 39), "30-39")
    .when(col("line_speed").between(40, 49), "40-49")
    .when(col("line_speed").between(50, 59), "50-59")
    .when(col("line_speed").between(60, 69), "60-69")
    .when(col("line_speed").between(70, 79), "70-79")
    .when(col("line_speed").between(80, 89), "80-89")
    .when(col("line_speed").between(90, 99), "90-99")
    .when(col("line_speed").between(100, 108), "100-108")
    .otherwise("Over 108")
).groupBy("line_speed_bucket").agg(
    count("*").alias("total_customers"),
    avg("churned").alias("churn_rate")
).withColumn("pct_of_total", col("total_customers") / total * 100).toPandas()

bucket_order = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59",
                "60-69", "70-79", "80-89", "90-99", "100-108", "Over 108", "Missing"]

line_speed_buckets['line_speed_bucket'] = pd.Categorical(
    line_speed_buckets['line_speed_bucket'], categories=bucket_order, ordered=True
)
line_speed_buckets = line_speed_buckets.sort_values('line_speed_bucket')

plt.figure(figsize=(12, 5))
plt.bar(range(len(line_speed_buckets)), line_speed_buckets['churn_rate'],
        color='steelblue', alpha=0.7)
plt.axhline(y=overall_churn_rate, color='red', linestyle='--',
            label=f'Overall churn ({overall_churn_rate:.1%})')
plt.xlabel('Line Speed Bucket (Mbps)')
plt.ylabel('Churn Rate')
plt.title('Churn Rate by Line Speed')
plt.xticks(range(len(line_speed_buckets)), line_speed_buckets['line_speed_bucket'],
           rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##8) sales channel

# COMMAND ----------

total = silver_raw_info.count()

# Original groups
sales_channel_original = silver_raw_info.groupBy("sales_channel").agg(
    count("*").alias("total_customers"),
    avg("churned").alias("churn_rate")
).withColumn("pct_of_total", col("total_customers") / total * 100) \
 .orderBy(col("total_customers").desc())

sales_channel_original.toPandas()


# COMMAND ----------

# Collapsed groups
sales_channel_collapsed = silver_raw_info.withColumn(
    "sales_channel_clean",
    when(col("sales_channel") == "Other", "Unknown")
    .when(col("sales_channel").isin(["Online - Search", "Online - Ambient"]), "Online - Organic")
    .when(col("sales_channel").isin(["Field", "Online - Other", "Outbound"]), "Other")
    .otherwise(col("sales_channel"))
).groupBy("sales_channel_clean").agg(
    count("*").alias("total_customers"),
    avg("churned").alias("churn_rate")
).withColumn("pct_of_total", col("total_customers") / total * 100) \
 .orderBy(col("total_customers").desc())

sales_channel_collapsed.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ##9) crm_package_name

# COMMAND ----------

print("=== CRM_PACKAGE_NAME ===")
print("\nValue counts and churn rates:")
silver_raw_info.groupBy("crm_package_name").agg(
    spark_count("*").alias("total_customers"),
    avg("churned").alias("churn_rate")
).withColumn("overall_churn_rate", lit(overall_churn_rate)) \
 .orderBy(col("total_customers").desc()).toPandas()

# COMMAND ----------

from pyspark.sql.functions import col, when, count as spark_count, avg, lit

# Create the mapping
crm_mapped = silver_raw_info.withColumn(
    "crm_package_clean",
    when(col("crm_package_name").isin(["Ultra Fibre Optic", "Ultra Fibre Optic Broadband"]), "Ultra Fibre Optic")
    .when(col("crm_package_name") == "Fibre 65 (FTTC-OR)", "Fibre 65 (FTTC-OR)")
    .when(col("crm_package_name") == "Fibre 35 (FTTC-OR)", "Fibre 35 (FTTC-OR)")
    .when(col("crm_package_name") == "Faster Fibre", "Faster Fibre")
    .when(col("crm_package_name") == "Fast Broadband", "Fast Broadband")
    .when(col("crm_package_name") == "Fibre 150 (GFast-OR)", "Fibre 150 (GFast-OR)")
    .when(col("crm_package_name") == "Broadband Only (SMPF)", "Broadband Only (SMPF)")
    .otherwise("Other")
)

# Get stats for new categories
new_categories = crm_mapped.groupBy("crm_package_clean").agg(
    spark_count("*").alias("total_customers"),
    avg("churned").alias("churn_rate")
).withColumn("overall_churn_rate", lit(overall_churn_rate)).toPandas()

# Get mapping of original to new categories
mapping = crm_mapped.select("crm_package_name", "crm_package_clean").distinct().toPandas()
mapping_grouped = mapping.groupby("crm_package_clean")["crm_package_name"].apply(lambda x: ", ".join(sorted(x))).reset_index()
mapping_grouped.columns = ["new_category", "original_categories"]

# Merge
final_table = new_categories.merge(mapping_grouped, left_on="crm_package_clean", right_on="new_category").drop("new_category", axis=1)
final_table = final_table[["crm_package_clean", "original_categories", "total_customers", "churn_rate", "overall_churn_rate"]]
final_table = final_table.sort_values("total_customers", ascending=False)

final_table

# COMMAND ----------

total = silver_raw_info.count()

crm_package = silver_raw_info.withColumn(
    "crm_package_clean",
    when(col("crm_package_name").isin(["Ultra Fibre Optic", "Ultra Fibre Optic Broadband"]), "Ultra Fibre Optic")
    .when(col("crm_package_name") == "Fibre 65 (FTTC-OR)", "Fibre 65 (FTTC-OR)")
    .when(col("crm_package_name") == "Fibre 35 (FTTC-OR)", "Fibre 35 (FTTC-OR)")
    .when(col("crm_package_name") == "Faster Fibre", "Faster Fibre")
    .when(col("crm_package_name") == "Fast Broadband", "Fast Broadband")
    .when(col("crm_package_name") == "Fibre 150 (GFast-OR)", "Fibre 150 (GFast-OR)")
    .when(col("crm_package_name") == "Broadband Only (SMPF)", "Broadband Only (SMPF)")
    .otherwise("Other")
).groupBy("crm_package_clean").agg(
    count("*").alias("total_customers"),
    avg("churned").alias("churn_rate")
).withColumn("pct_of_total", col("total_customers") / total * 100) \
 .orderBy(col("total_customers").desc())

crm_package.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ##10) Tenure

# COMMAND ----------

print("=== TENURE_DAYS ===")
print("\nSummary statistics:")
silver_raw_info.select("tenure_days").summary().show()

print("\nDistinct values count:")
print(f"Number of distinct tenure_days values: {silver_raw_info.select('tenure_days').distinct().count()}")

# Look at 10 decile buckets for EDA
quantiles = silver_raw_info.approxQuantile("tenure_days", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 0.01)
print(f"\nDecile boundaries: {quantiles}")

# Create buckets for visualization
tenure_buckets = silver_raw_info.withColumn(
    "tenure_bucket",
    when(col("tenure_days").isNull(), "Missing")
    .when(col("tenure_days") <= quantiles[0], f"0-{quantiles[0]:.0f}")
    .when(col("tenure_days") <= quantiles[1], f"{quantiles[0]:.0f}-{quantiles[1]:.0f}")
    .when(col("tenure_days") <= quantiles[2], f"{quantiles[1]:.0f}-{quantiles[2]:.0f}")
    .when(col("tenure_days") <= quantiles[3], f"{quantiles[2]:.0f}-{quantiles[3]:.0f}")
    .when(col("tenure_days") <= quantiles[4], f"{quantiles[3]:.0f}-{quantiles[4]:.0f}")
    .when(col("tenure_days") <= quantiles[5], f"{quantiles[4]:.0f}-{quantiles[5]:.0f}")
    .when(col("tenure_days") <= quantiles[6], f"{quantiles[5]:.0f}-{quantiles[6]:.0f}")
    .when(col("tenure_days") <= quantiles[7], f"{quantiles[6]:.0f}-{quantiles[7]:.0f}")
    .when(col("tenure_days") <= quantiles[8], f"{quantiles[7]:.0f}-{quantiles[8]:.0f}")
    .otherwise(f"{quantiles[8]:.0f}+")
).groupBy("tenure_bucket").agg(
    spark_count("*").alias("total_customers"),
    avg("churned").alias("churn_rate")
).toPandas()

print("\nChurn rate by tenure decile:")
tenure_buckets

# COMMAND ----------

print("=== TENURE_DAYS ===")

# Get deciles
quantiles = silver_raw_info.approxQuantile("tenure_days", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 0.01)
print(f"Decile boundaries: {quantiles}")

# Create buckets
tenure_buckets = silver_raw_info.withColumn(
    "tenure_bucket",
    when(col("tenure_days").isNull(), "Missing")
    .when(col("tenure_days") <= quantiles[0], f"0-{quantiles[0]:.0f}")
    .when(col("tenure_days") <= quantiles[1], f"{quantiles[0]:.0f}-{quantiles[1]:.0f}")
    .when(col("tenure_days") <= quantiles[2], f"{quantiles[1]:.0f}-{quantiles[2]:.0f}")
    .when(col("tenure_days") <= quantiles[3], f"{quantiles[2]:.0f}-{quantiles[3]:.0f}")
    .when(col("tenure_days") <= quantiles[4], f"{quantiles[3]:.0f}-{quantiles[4]:.0f}")
    .when(col("tenure_days") <= quantiles[5], f"{quantiles[4]:.0f}-{quantiles[5]:.0f}")
    .when(col("tenure_days") <= quantiles[6], f"{quantiles[5]:.0f}-{quantiles[6]:.0f}")
    .when(col("tenure_days") <= quantiles[7], f"{quantiles[6]:.0f}-{quantiles[7]:.0f}")
    .when(col("tenure_days") <= quantiles[8], f"{quantiles[7]:.0f}-{quantiles[8]:.0f}")
    .otherwise(f"{quantiles[8]:.0f}+")
).groupBy("tenure_bucket").agg(
    spark_count("*").alias("total_customers"),
    avg("churned").alias("churn_rate")
).toPandas()

# Define proper order
bucket_order = [
    f"0-{quantiles[0]:.0f}",
    f"{quantiles[0]:.0f}-{quantiles[1]:.0f}",
    f"{quantiles[1]:.0f}-{quantiles[2]:.0f}",
    f"{quantiles[2]:.0f}-{quantiles[3]:.0f}",
    f"{quantiles[3]:.0f}-{quantiles[4]:.0f}",
    f"{quantiles[4]:.0f}-{quantiles[5]:.0f}",
    f"{quantiles[5]:.0f}-{quantiles[6]:.0f}",
    f"{quantiles[6]:.0f}-{quantiles[7]:.0f}",
    f"{quantiles[7]:.0f}-{quantiles[8]:.0f}",
    f"{quantiles[8]:.0f}+",
    "Missing"
]

tenure_buckets['tenure_bucket'] = pd.Categorical(
    tenure_buckets['tenure_bucket'], 
    categories=bucket_order, 
    ordered=True
)
tenure_buckets = tenure_buckets.sort_values('tenure_bucket')



# Plot
plt.figure(figsize=(12, 6))
plt.bar(range(len(tenure_buckets)), tenure_buckets['churn_rate'], color='steelblue', alpha=0.7)
plt.axhline(y=overall_churn_rate, color='red', linestyle='--', label=f'Overall churn ({overall_churn_rate:.1%})')
plt.xlabel('Tenure Days Bucket')
plt.ylabel('Churn Rate')
plt.title('Churn Rate by Tenure (Days with Company)')
plt.xticks(range(len(tenure_buckets)), tenure_buckets['tenure_bucket'], rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# COMMAND ----------


# Plot
plt.figure(figsize=(12, 6))
plt.bar(range(len(tenure_buckets)), tenure_buckets['churn_rate'], color='steelblue', alpha=0.7)
plt.axhline(y=overall_churn_rate, color='red', linestyle='--', label=f'Overall churn ({overall_churn_rate:.1%})')
plt.xlabel('Tenure Days Bucket')
plt.ylabel('Churn Rate')
plt.title('Churn Rate by Tenure (Days with Company)')
plt.xticks(range(len(tenure_buckets)), tenure_buckets['tenure_bucket'], rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # 11) Calls

# COMMAND ----------

# Get distinct customers who made calls
customers_with_calls = calls_df.select("unique_customer_identifier").distinct()

analysis = silver_raw_info.select("unique_customer_identifier", "churned").join(
    customers_with_calls,
    on="unique_customer_identifier",
    how="left"
).withColumn(
    "has_calls",
    when(customers_with_calls.unique_customer_identifier.isNotNull(), 1).otherwise(0)
)

# Convert to pandas and show churn by has_calls
analysis_pd = analysis.toPandas()

summary = analysis_pd.groupby('has_calls').agg({
    'churned': 'mean',
    'unique_customer_identifier': 'count'
}).round(3)
summary.columns = ['churn_rate', 'customer_count']
summary = summary.reset_index()
summary['churn_rate'] = (summary['churn_rate'] * 100).round(1)

print("=== CHURN RATE BY CALLS ===")
print(summary)

# COMMAND ----------

# MAGIC %md
# MAGIC # 12) Usage

# COMMAND ----------

# Join usage data with silver table to filter by behav_win month
silver_raw_info = spark.table("churn_silver.raw_info")
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
display(filtered_usage.limit(10))

# Save as a reduced table for easier processing
filtered_usage.write.format("delta").mode("overwrite").saveAsTable("churn_silver.usage_behav_month")
print("Saved to churn_silver.usage_behav_month")

# COMMAND ----------

from pyspark.sql.functions import col, when
import pandas as pd

# Get customers with usage data
silver_raw_info = spark.table("churn_silver.raw_info")
usage_behav_month = spark.table("churn_silver.usage_behav_month")

customers_with_usage = usage_behav_month.select("unique_customer_identifier").distinct()

# Add flag for has_usage
analysis = silver_raw_info.join(
    customers_with_usage,
    on="unique_customer_identifier",
    how="left"
).withColumn(
    "has_usage",
    when(customers_with_usage.unique_customer_identifier.isNotNull(), 1).otherwise(0)
)

# Convert to pandas
analysis_pd = analysis.select("unique_customer_identifier", "churned", "has_usage").toPandas()

# Cross-tabulation
print("=== USAGE DATA vs CHURN ===")
crosstab = pd.crosstab(analysis_pd['has_usage'], analysis_pd['churned'], margins=True)
print(crosstab)

# Churn rates
print("\n=== CHURN RATE BY USAGE AVAILABILITY ===")
summary = analysis_pd.groupby('has_usage').agg({
    'churned': ['mean', 'count']
}).round(3)
summary

# COMMAND ----------

# Get DISTINCT usage days per customer
usage_behav_month = spark.table("churn_silver.usage_behav_month")
days_per_customer = usage_behav_month.groupBy("unique_customer_identifier").agg(
    countDistinct("calendar_date").alias("usage_days")
)

# Join with raw_info and calculate days in behav_win month
silver_raw_info = spark.table("churn_silver.raw_info")

analysis = silver_raw_info.select("unique_customer_identifier", "churned", "behav_win").join(
    days_per_customer,
    on="unique_customer_identifier",
    how="left"
).fillna(0, subset=["usage_days"])

# Calculate missed days
analysis = analysis.withColumn(
    "days_in_month",
    dayofmonth(last_day(col("behav_win")))
).withColumn(
    "missed_days",
    col("days_in_month") - col("usage_days")
)

# Create buckets in PySpark
analysis = analysis.withColumn(
    "usage_bucket",
    when(col("usage_days") == 0, "no_usage")
    .when(col("missed_days") == 0, "full_usage")
    .when((col("missed_days") >= 1) & (col("missed_days") <= 14), "missed_1_to_14")
    .otherwise("missed_15_plus")
)

# Convert to pandas
analysis_pd = analysis.toPandas()

# Create two summary tables
summary1 = analysis_pd.groupby('usage_bucket').agg({
    'churned': 'mean',
    'unique_customer_identifier': 'count'
}).round(3)
summary1.columns = ['churn_rate', 'customer_count']
summary1['churn_rate'] = (summary1['churn_rate'] * 100).round(1)

# COMMAND ----------

# Sum total download usage per customer in behav_win month
usage_behav_month = spark.table("churn_silver.usage_behav_month")

total_usage_per_customer = usage_behav_month.groupBy("unique_customer_identifier").agg(
    spark_sum("usage_download_mbs").alias("total_download_mbs")
)

# Join with raw_info to get churned flag
silver_raw_info = spark.table("churn_silver.raw_info")

analysis = silver_raw_info.select("unique_customer_identifier", "churned").join(
    total_usage_per_customer,
    on="unique_customer_identifier",
    how="left"
).fillna(0, subset=["total_download_mbs"])

# Convert to pandas
analysis_pd = analysis.toPandas()

# Create 10 buckets (deciles)
analysis_pd['usage_bucket'] = pd.qcut(analysis_pd['total_download_mbs'], q=10, labels=False, duplicates='drop')

# Calculate churn rate per bucket
churn_by_bucket = analysis_pd.groupby('usage_bucket').agg({
    'churned': ['mean', 'count'],
    'total_download_mbs': ['min', 'max']
}).round(3)

churn_by_bucket.columns = ['churn_rate', 'customer_count', 'min_usage_mb', 'max_usage_mb']
churn_by_bucket = churn_by_bucket.reset_index()
churn_by_bucket['churn_rate'] = (churn_by_bucket['churn_rate'] * 100).round(1)

print("=== CHURN RATE BY DOWNLOAD USAGE BUCKET ===")
print(churn_by_bucket)

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.bar(churn_by_bucket['usage_bucket'], churn_by_bucket['churn_rate'])
plt.xlabel('Usage Bucket (0=lowest, 9=highest)')
plt.ylabel('Churn Rate (%)')
plt.title('Churn Rate by Download Usage Decile')
plt.show()

# COMMAND ----------

# Sum total upload usage per customer in behav_win month
usage_behav_month = spark.table("churn_silver.usage_behav_month")

total_upload_per_customer = usage_behav_month.groupBy("unique_customer_identifier").agg(
    spark_sum("usage_upload_mbs").alias("total_upload_mbs")
)

# Join with raw_info to get churned flag
silver_raw_info = spark.table("churn_silver.raw_info")

analysis = silver_raw_info.select("unique_customer_identifier", "churned").join(
    total_upload_per_customer,
    on="unique_customer_identifier",
    how="left"
).fillna(0, subset=["total_upload_mbs"])

# Convert to pandas
analysis_pd = analysis.toPandas()

# Create 10 buckets (deciles)
analysis_pd['usage_bucket'] = pd.qcut(analysis_pd['total_upload_mbs'], q=10, labels=False, duplicates='drop')

# Calculate churn rate per bucket
churn_by_bucket = analysis_pd.groupby('usage_bucket').agg({
    'churned': ['mean', 'count'],
    'total_upload_mbs': ['min', 'max']
}).round(3)

churn_by_bucket.columns = ['churn_rate', 'customer_count', 'min_usage_mb', 'max_usage_mb']
churn_by_bucket = churn_by_bucket.reset_index()
churn_by_bucket['churn_rate'] = (churn_by_bucket['churn_rate'] * 100).round(1)

print("=== CHURN RATE BY UPLOAD USAGE BUCKET ===")
print(churn_by_bucket)

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.bar(churn_by_bucket['usage_bucket'], churn_by_bucket['churn_rate'])
plt.xlabel('Usage Bucket (0=lowest, 9=highest)')
plt.ylabel('Churn Rate (%)')
plt.title('Churn Rate by Upload Usage Decile')
plt.show()

# COMMAND ----------


