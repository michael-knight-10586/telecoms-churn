# Databricks notebook source
from pyspark.sql.functions import col, count, min, max, to_date
import matplotlib.pyplot as plt
import pandas as pd

# Use the bronze database where our tables are stored
spark.sql("USE churn_bronze")
print("Connected to churn_bronze database")

# COMMAND ----------

# Load the cease data and examine its structure
cease_df = spark.table("cease_bronze")

# Show the schema - column names and data types
cease_df.printSchema()

# Show basic statistics
print(f"\nTotal rows: {cease_df.count()}")
print(f"Total columns: {len(cease_df.columns)}")

# COMMAND ----------

# Get min and max dates
date_range = cease_df.select(
    min("cease_placed_date").alias("min_date"),
    max("cease_placed_date").alias("max_date")
).collect()[0]

print(f"Cease data time period:")
print(f"Earliest cease: {date_range['min_date']}")
print(f"Latest cease: {date_range['max_date']}")

# Group by cease_placed_date and count entries
cease_by_date = cease_df.groupBy("cease_placed_date").agg(
    count("*").alias("cease_count")
).orderBy("cease_placed_date")

# Convert to pandas for plotting
cease_pd = cease_by_date.toPandas()
cease_pd['cease_placed_date'] = pd.to_datetime(cease_pd['cease_placed_date'])

# Create the plot
plt.figure(figsize=(14, 6))
plt.plot(cease_pd['cease_placed_date'], cease_pd['cease_count'], linewidth=1)
plt.xlabel('Date')
plt.ylabel('Number of Cease Requests')
plt.title('Cease Requests Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True, alpha=0.3)
plt.show()

# COMMAND ----------

from pyspark.sql.functions import countDistinct, count, min, max, col
import matplotlib.pyplot as plt
import pandas as pd

customer_info_df = spark.read.parquet(
    "dbfs:/FileStore/tables/customer_info.parquet"
)

# Basic counts
total_rows = customer_info_df.count()
unique_customers = customer_info_df.select(countDistinct("unique_customer_identifier")).collect()[0][0]

print(f"Customer Info Table Statistics:")
print(f"Total rows: {total_rows:,}")
print(f"Unique customers: {unique_customers:,}")
print(f"Average snapshots per customer: {total_rows/unique_customers:.2f}")

# Get time period
date_range = customer_info_df.select(
    min("datevalue").alias("min_date"),
    max("datevalue").alias("max_date"),
    countDistinct("datevalue").alias("unique_months")
).collect()[0]

print(f"\nTime period:")
print(f"Earliest snapshot: {date_range['min_date']}")
print(f"Latest snapshot: {date_range['max_date']}")
print(f"Number of unique months: {date_range['unique_months']}")

# Count customers per month
customers_by_month = customer_info_df.groupBy("datevalue").agg(
    count("unique_customer_identifier").alias("customer_count")
).orderBy("datevalue")

# Convert to pandas and plot
customers_pd = customers_by_month.toPandas()
customers_pd['datevalue'] = pd.to_datetime(customers_pd['datevalue'])

plt.figure(figsize=(14, 6))
plt.plot(customers_pd['datevalue'], customers_pd['customer_count'], marker='o', linewidth=2)
plt.xlabel('Month')
plt.ylabel('Number of Customers')
plt.title('Customer Base Size Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True, alpha=0.3)
plt.show()

# COMMAND ----------

 # Get the detailed month-by-month breakdown
customers_pd_detailed = customers_by_month.toPandas()
customers_pd_detailed['datevalue'] = pd.to_datetime(customers_pd_detailed['datevalue'])

print(customers_pd_detailed)

# Check if the decline rate is consistent
customers_pd_detailed['month_over_month_change'] = customers_pd_detailed['customer_count'].diff()
customers_pd_detailed['percent_change'] = (customers_pd_detailed['month_over_month_change'] / customers_pd_detailed['customer_count'].shift(1) * 100)

print("\nMonth over month changes:")
print(customers_pd_detailed[['datevalue', 'customer_count', 'month_over_month_change', 'percent_change']])

# COMMAND ----------

first_month = customer_info_df.filter(col("datevalue") == date_range['min_date'])
last_month = customer_info_df.filter(col("datevalue") == date_range['max_date'])

first_month_customers = set([row.unique_customer_identifier for row in first_month.select("unique_customer_identifier").collect()])
last_month_customers = set([row.unique_customer_identifier for row in last_month.select("unique_customer_identifier").collect()])

customers_only_in_first = len(first_month_customers - last_month_customers)
customers_only_in_last = len(last_month_customers - first_month_customers)
customers_in_both = len(first_month_customers.intersection(last_month_customers))

print(f"\nCustomer movement between first and last month:")
print(f"Only in first month: {customers_only_in_first:,}")
print(f"Only in last month: {customers_only_in_last:,}")
print(f"In both months: {customers_in_both:,}")

# COMMAND ----------

from pyspark.sql.functions import countDistinct, count

# Check if customer-month combinations are unique
total_rows = customer_info_df.count()
unique_customer_months = customer_info_df.select(
    countDistinct("unique_customer_identifier", "datevalue")
).collect()[0][0]

print(f"Customer Info Grain Check:")
print(f"Total rows: {total_rows:,}")
print(f"Unique customer-month combinations: {unique_customer_months:,}")

if total_rows == unique_customer_months:
    print("✓ Table grain is correct - one row per customer per month")
else:
    print("✗ WARNING: Duplicate customer-month records exist!")
    print(f"Number of duplicates: {total_rows - unique_customer_months:,}")

# Also check how many times each customer appears on average
customer_frequency = customer_info_df.groupBy("unique_customer_identifier").agg(
    count("*").alias("months_appearing")
)

avg_months = customer_frequency.agg({"months_appearing": "avg"}).collect()[0][0]
print(f"\nAverage months per customer: {avg_months:.2f}")

# COMMAND ----------

from pyspark.sql.functions import countDistinct

# Get unique customer count
unique_customers = customer_info_df.select(countDistinct("unique_customer_identifier")).collect()[0][0]
total_rows = customer_info_df.count()

print(f"Unique customers: {unique_customers:,}")
print(f"Total rows: {total_rows:,}")

# COMMAND ----------

from pyspark.sql.functions import col, weekofyear, year, concat, lit, count, sum as spark_sum, min as spark_min, max as spark_max
import matplotlib.pyplot as plt
import pandas as pd

# Get weekly churn counts
cease_weekly = cease_df.withColumn("year", year("cease_placed_date")) \
    .withColumn("week", weekofyear("cease_placed_date")) \
    .withColumn("year_week", concat(col("year"), lit("-W"), col("week"))) \
    .groupBy("year_week", "cease_placed_date") \
    .agg(count("*").alias("churns")) \
    .groupBy("year_week") \
    .agg(
        count("*").alias("days_in_week"),
        spark_sum("churns").alias("weekly_churns")
    ) \
    .orderBy("year_week")

# Convert to pandas
weekly_pd = cease_weekly.toPandas()

# Get the date range and starting customer count
min_date = cease_df.select(spark_min("cease_placed_date")).collect()[0][0]
max_date = cease_df.select(spark_max("cease_placed_date")).collect()[0][0]

# Get starting active customers from first month of customer_info
first_month_count = customer_info_df.filter(
    col("datevalue") == customer_info_df.select(spark_min("datevalue")).collect()[0][0]
).count()

print(f"Starting active customers: {first_month_count:,}")
print(f"Date range: {min_date} to {max_date}")

# Calculate cumulative churns and active customers
weekly_pd = weekly_pd.sort_values('year_week')
weekly_pd['cumulative_churns'] = weekly_pd['weekly_churns'].cumsum()
weekly_pd['active_customers'] = first_month_count - weekly_pd['cumulative_churns']

# Plot
fig, ax1 = plt.subplots(figsize=(16, 7))

x_positions = range(len(weekly_pd))

# Plot active customers on left axis
ax1.plot(x_positions, weekly_pd['active_customers'], color='steelblue', linewidth=2, label='Active Customers')
ax1.set_xlabel('Week')
ax1.set_ylabel('Active Customers', color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')
ax1.grid(True, alpha=0.3)

# Plot weekly churns on right axis
ax2 = ax1.twinx()
ax2.bar(x_positions, weekly_pd['weekly_churns'], color='coral', alpha=0.6, label='Weekly Churns')
ax2.set_ylabel('Weekly Churns', color='coral')
ax2.tick_params(axis='y', labelcolor='coral')

# Set x-axis labels (sho

# COMMAND ----------

# Load the correctly uploaded usage file
usage_df = spark.read.parquet("dbfs:/FileStore/tables/usage_new.parquet")

# Show the columns
print("Usage table columns:")
print(usage_df.columns)

# Get schema for more details
print("\nUsage table schema:")
usage_df.printSchema()

# Get unique customers and total rows
from pyspark.sql.functions import countDistinct

total_rows = usage_df.count()
unique_customers = usage_df.select(countDistinct("unique_customer_identifier")).collect()[0][0]

print(f"\nUsage Table Statistics:")
print(f"Total rows: {total_rows:,}")
print(f"Unique customers: {unique_customers:,}")
print(f"Average records per customer: {total_rows/unique_customers:.2f}")

# COMMAND ----------

from pyspark.sql.functions import max as spark_max, min as spark_min, datediff, col

# For customers who churned, compare their last usage date to their cease date
churned_customers_usage = usage_df.join(
    cease_df.select("unique_customer_identifier", "cease_placed_date"),
    on="unique_customer_identifier",
    how="inner"
).groupBy("unique_customer_identifier", "cease_placed_date").agg(
    spark_max("calendar_date").alias("last_usage_date"),
    spark_min("calendar_date").alias("first_usage_date")
)

# Calculate the difference between last usage and cease date
churned_customers_usage = churned_customers_usage.withColumn(
    "days_diff_usage_to_cease",
    datediff(col("cease_placed_date"), col("last_usage_date"))
)

# Look at the distribution
churned_customers_usage.select("days_diff_usage_to_cease").summary().show()

# Show some examples
print("Sample of churned customers - last usage date vs cease date:")
churned_customers_usage.select(
    "unique_customer_identifier", 
    "first_usage_date",
    "last_usage_date", 
    "cease_placed_date",
    "days_diff_usage_to_cease"
).show(20, truncate=False)

# COMMAND ----------

from pyspark.sql.functions import countDistinct

# Join usage with cease and find usage after cease completion
usage_after_cease = usage_df.join(
    cease_df.select("unique_customer_identifier", "cease_completed_date"),
    on="unique_customer_identifier",
    how="inner"
).filter(col("calendar_date") > col("cease_completed_date"))

# Count unique customers with usage after cease
customers_with_post_cease_usage = usage_after_cease.select(
    countDistinct("unique_customer_identifier")
).collect()[0][0]

print(f"Customers with usage after cease_completed_date: {customers_with_post_cease_usage:,}")

# Also check how many cease records have null cease_completed_date
null_completed = cease_df.filter(col("cease_completed_date").isNull()).count()
total_cease = cease_df.count()

print(f"\nCease records with null cease_completed_date: {null_completed:,} out of {total_cease:,}")
print(f"Percentage not completed: {null_completed/total_cease*100:.1f}%")

# COMMAND ----------

from pyspark.sql.functions import countDistinct

# Join usage with cease and find usage after cease was PLACED
usage_after_cease_placed = usage_df.join(
    cease_df.select("unique_customer_identifier", "cease_placed_date"),
    on="unique_customer_identifier",
    how="inner"
).filter(col("calendar_date") > col("cease_placed_date"))

# Count customers and total records
customers_with_post_placed_usage = usage_after_cease_placed.select(
    countDistinct("unique_customer_identifier")
).collect()[0][0]

total_post_placed_records = usage_after_cease_placed.count()

print(f"Customers with usage after cease_placed_date: {customers_with_post_placed_usage:,}")
print(f"Total usage records after cease_placed_date: {total_post_placed_records:,}")

# COMMAND ----------

from pyspark.sql.functions import countDistinct

# Read customer_info.parquet from DBFS
# Parquet files don't need header or inferSchema - metadata is built in
customer_info_df = spark.read.parquet(
    "dbfs:/FileStore/tables/customer_info.parquet"
)

unique_customers_info = customer_info_df.select(countDistinct("unique_customer_identifier")).collect()[0][0]

print(f"Unique customers in customer_info: {unique_customers_info:,}")

# COMMAND ----------

unique_customers_usage = usage_df.select(countDistinct("unique_customer_identifier")).collect()[0][0]

print(f"Unique customers in usage: {unique_customers_usage:,}")

# COMMAND ----------

customers_not_in_usage = customer_info_df.select("unique_customer_identifier").distinct() \
    .join(
        usage_df.select("unique_customer_identifier").distinct(),
        on="unique_customer_identifier",
        how="left_anti"
    )

count_not_in_usage = customers_not_in_usage.count()
print(f"Customers in customer_info but not in usage: {count_not_in_usage:,}")

# COMMAND ----------

# Get full customer_info data for customers not in usage
customers_not_in_usage_full = customer_info_df.join(
    usage_df.select("unique_customer_identifier").distinct(),
    on="unique_customer_identifier",
    how="left_anti"
)

# Join with cease data to get all columns from both tables
full_data = customers_not_in_usage_full.join(
    cease_df,
    on="unique_customer_identifier",
    how="left"
)

# Convert to pandas and save
full_data_pd = full_data.toPandas()
full_data_pd.to_csv('/dbfs/FileStore/tables/customers_not_in_usage_full2.csv', index=False)

print(f"Saved {len(full_data_pd)} rows with all columns to customers_not_in_usage_full2.csv")
print(f"Columns: {list(full_data_pd.columns)}")

# COMMAND ----------

full_data_pd.sample()

# COMMAND ----------

from pyspark.sql.functions import col, count, when, isnan, isnull
from pyspark.sql.functions import countDistinct

# Read customer_info.parquet from DBFS
# Parquet files don't need header or inferSchema - metadata is built in
customer_info_df = spark.read.parquet(
    "dbfs:/FileStore/tables/customer_info.parquet"
)

from pyspark.sql.functions import col, count, when, isnan, isnull
from pyspark.sql.types import DoubleType, FloatType

# Get total rows
total_rows = customer_info_df.count()
print(f"Total rows: {total_rows:,}\n")

# Check missing values - only use isnan for numeric types
print("Missing values by column:")
missing_counts = []
for c in customer_info_df.columns:
    col_type = dict(customer_info_df.dtypes)[c]
    if col_type in ['double', 'float']:
        missing_counts.append(count(when(col(c).isNull() | isnan(c), c)).alias(c))
    else:
        missing_counts.append(count(when(col(c).isNull(), c)).alias(c))

customer_info_df.select(missing_counts).show(vertical=True)

# Show schema
print("\nSchema:")
customer_info_df.printSchema()

# Check distinct values for categorical columns
categorical_cols = ['contract_status', 'technology', 'sales_channel', 'crm_package_name']

print("\nDistinct value counts:")
for col_name in categorical_cols:
    distinct_count = customer_info_df.select(col_name).distinct().count()
    print(f"{col_name}: {distinct_count} distinct values")

# Show distributions
print("\nContract Status distribution:")
customer_info_df.groupBy("contract_status").count().orderBy(col("count").desc()).show()

print("\nTechnology distribution:")
customer_info_df.groupBy("technology").count().orderBy(col("count").desc()).show()

print("\nSales Channel distribution:")
customer_info_df.groupBy("sales_channel").count().orderBy(col("count").desc()).show()

# COMMAND ----------

# Filter for rows where ooc_days is null
missing_ooc = customer_info_df.filter(col("ooc_days").isNull())

# Show count
missing_count = missing_ooc.count()
print(f"Rows with missing ooc_days: {missing_count:,}\n")

# Show sample of all columns
missing_ooc.show(20, truncate=False)

# COMMAND ----------


