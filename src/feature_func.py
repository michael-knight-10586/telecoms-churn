"""
Feature engineering functions for customer info data
"""

from pyspark.sql.functions import col, when, DataFrame
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    countDistinct, dayofmonth, last_day, when, 
    col, lit, year, month, sum as spark_sum
)



def create_contract_status_features(spark) -> DataFrame:
    """
    Creates one-hot encoded features for contract_status
    
    Returns:
        DataFrame with unique_customer_identifier and contract_status binary columns
    """
    # Read raw_info table
    silver_raw_info = spark.table("churn_silver.raw_info")
    
    # Get unique contract statuses
    statuses = silver_raw_info.select("contract_status").distinct().rdd.flatMap(lambda x: x).collect()
    
    # Start with customer ID and contract_status
    result = silver_raw_info.select("unique_customer_identifier", "contract_status")
    
    # Create binary column for each status
    for status in statuses:
        # Clean column name
        col_name = f"contract_status_{status.replace(' ', '_').replace('/', '_').lower()}"
        result = result.withColumn(col_name, when(col("contract_status") == status, 1).otherwise(0))
    
    # Drop original contract_status column
    result = result.drop("contract_status")
    
    return result


def create_has_calls_feature(spark):
    """
    For each customer in raw_info, check if they have calls in their behav_win month
    Returns: DataFrame with unique_customer_identifier and has_calls (0/1)
    """
    # Read tables
    silver_raw_info = spark.table("churn_silver.raw_info")
    calls_df = spark.table("churn_bronze.calls_bronze")

    # Start with all customers and their behav_win
    customers = silver_raw_info.select("unique_customer_identifier", "behav_win")
    
    print(f"Total customers in raw_info: {customers.count()}")
    
    # Join with calls table
    customer_calls = customers.join(
        calls_df,
        on="unique_customer_identifier",
        how="left"
    )
    
    # Filter to calls that occurred in the behav_win month
    calls_in_behav_month = customer_calls.filter(
        (year(col("event_date")) == year(col("behav_win"))) &
        (month(col("event_date")) == month(col("behav_win")))
    )
    
    # Get distinct customers who made calls
    customers_with_calls = calls_in_behav_month.select("unique_customer_identifier").distinct().withColumn("has_calls", lit(1))
    
    print(f"Customers with calls in behav_win month: {customers_with_calls.count()}")
    
    # Join back to all customers
    result = customers.select("unique_customer_identifier").join(
        customers_with_calls,
        on="unique_customer_identifier",
        how="left"
    ).fillna(0, subset=["has_calls"])
    
    return result


def create_usage_days_feature(spark) -> DataFrame:
    """
    Creates bucketed usage_days feature based on missed days in behav_win month
    One-hot encoded into 4 binary columns
    
    Returns:
        DataFrame with unique_customer_identifier and binary usage_days_bucket columns
    """
    # Read tables
    silver_raw_info = spark.table("churn_silver.raw_info")
    usage_behav_month = spark.table("churn_silver.usage_behav_month")
    
    # Count distinct usage days per customer
    days_per_customer = usage_behav_month.groupBy("unique_customer_identifier").agg(
        countDistinct("calendar_date").alias("usage_days")
    )
    
    # Join with raw_info and calculate days in behav_win month
    result = silver_raw_info.select("unique_customer_identifier", "behav_win").join(
        days_per_customer,
        on="unique_customer_identifier",
        how="left"
    ).fillna(0, subset=["usage_days"])
    
    # Calculate missed days
    result = result.withColumn(
        "days_in_month",
        dayofmonth(last_day(col("behav_win")))
    ).withColumn(
        "missed_days",
        col("days_in_month") - col("usage_days")
    )
    
    # Create binary columns for each bucket (mutually exclusive)
    result = result.withColumn(
        "usage_days_bucket_no_usage",
        when(col("usage_days") == 0, 1).otherwise(0)
    ).withColumn(
        "usage_days_bucket_full_usage",
        when((col("missed_days") == 0) & (col("usage_days") > 0), 1).otherwise(0)
    ).withColumn(
        "usage_days_bucket_missed_1_to_14",
        when((col("missed_days") >= 1) & (col("missed_days") <= 14), 1).otherwise(0)
    ).withColumn(
        "usage_days_bucket_missed_15_plus",
        when((col("missed_days") > 14) & (col("usage_days") > 0), 1).otherwise(0)
    )
    
    # Select only ID and binary columns
    result = result.select(
        "unique_customer_identifier",
        "usage_days_bucket_no_usage",
        "usage_days_bucket_full_usage",
        "usage_days_bucket_missed_1_to_14",
        "usage_days_bucket_missed_15_plus"
    )
    
    return result

def create_crm_package_features(spark) -> DataFrame:
    """
    Combine small crm_package_name categories and one-hot encode
    
    Returns:
        DataFrame with unique_customer_identifier and binary crm_package columns
    """
    # Read raw_info table
    silver_raw_info = spark.table("churn_silver.raw_info")
    result = silver_raw_info.select("unique_customer_identifier", "crm_package_name")
    
    # Define categories to keep (>450 customers)
    keep_categories = [
        "Fibre 65 (FTTC-OR)",
        "Fibre 35 (FTTC-OR)", 
        "Faster Fibre",
        "Fast Broadband",
        "Fibre 150 (GFast-OR)",
        "Broadband Only (SMPF)"
    ]
    
    # Combine categories
    result = result.withColumn(
        "crm_package_clean",
        when(col("crm_package_name").isin(["Ultra Fibre Optic", "Ultra Fibre Optic Broadband"]), "Ultra Fibre Optic")
        .when(col("crm_package_name").isin(keep_categories), col("crm_package_name"))
        .otherwise("Other")
    )
    
    # Get unique values after combining
    packages = result.select("crm_package_clean").distinct().rdd.flatMap(lambda x: x).collect()
    
    # Create binary column for each package
    for package in packages:
        if package is not None:
            col_name = f"crm_package_{package.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').lower()}"
            result = result.withColumn(col_name, when(col("crm_package_clean") == package, 1).otherwise(0))
    
    # Drop original columns
    result = result.drop("crm_package_name", "crm_package_clean")
    
    return result

def create_sales_channel_features(spark) -> DataFrame:
    """
    Combine small sales_channel categories and one-hot encode
    
    Returns:
        DataFrame with unique_customer_identifier and binary sales_channel columns
    """
    # Read raw_info table
    silver_raw_info = spark.table("churn_silver.raw_info")
    result = silver_raw_info.select("unique_customer_identifier", "sales_channel")
    
    # Combine categories
    result = result.withColumn(
        "sales_channel_clean",
        when(col("sales_channel") == "Other", "Unknown")
        .when(col("sales_channel").isin(["Online - Search", "Online - Ambient"]), "Online - Organic")
        .when(col("sales_channel").isin(["Field", "Online - Other", "Outbound"]), "Other")
        .otherwise(col("sales_channel"))
    )
    
    # Get unique values after combining
    channels = result.select("sales_channel_clean").distinct().rdd.flatMap(lambda x: x).collect()
    
    # Create binary column for each channel
    for channel in channels:
        if channel is not None:
            col_name = f"sales_channel_{channel.replace(' ', '_').replace('-', '').lower()}"
            result = result.withColumn(col_name, when(col("sales_channel_clean") == channel, 1).otherwise(0))
    
    # Drop original columns
    result = result.drop("sales_channel", "sales_channel_clean")
    
    return result

def create_speed_features(spark) -> DataFrame:
    """
    One-hot encode speed column (package speed, not line speed)
    
    Returns:
        DataFrame with unique_customer_identifier and binary package_speed columns
    """
    # Read raw_info table
    silver_raw_info = spark.table("churn_silver.raw_info")
    result = silver_raw_info.select("unique_customer_identifier", "speed")
    
    # Get unique speed values
    speeds = result.select("speed").distinct().rdd.flatMap(lambda x: x).collect()
    
    # Create binary column for each speed tier
    for speed in speeds:
        if speed is not None:
            col_name = f"package_speed_{speed}"
            result = result.withColumn(col_name, when(col("speed") == speed, 1).otherwise(0))
    
    # Drop original column
    result = result.drop("speed")
    
    return result

def create_technology_features(spark) -> DataFrame:
    """
    One-hot encode technology column
    
    Returns:
        DataFrame with unique_customer_identifier and binary technology columns
    """
    # Read raw_info table
    silver_raw_info = spark.table("churn_silver.raw_info")
    result = silver_raw_info.select("unique_customer_identifier", "technology")
    
    # Get unique technology values
    technologies = result.select("technology").distinct().rdd.flatMap(lambda x: x).collect()
    
    # Create binary column for each technology
    for tech in technologies:
        if tech is not None:
            col_name = f"technology_{tech.lower()}"
            result = result.withColumn(col_name, when(col("technology") == tech, 1).otherwise(0))
    
    # Drop original column
    result = result.drop("technology")
    
    return result

def create_ooc_days_features(spark) -> DataFrame:
    """
    Create OOC days bucket features
    
    Returns:
        DataFrame with unique_customer_identifier and binary bucket columns
    """
    # Read raw_info table
    silver_raw_info = spark.table("churn_silver.raw_info")
    result = silver_raw_info.select("unique_customer_identifier", "ooc_days")
    
    result = result.withColumn(
        "ooc_bucket_under_minus100",
        when(col("ooc_days") < -100, 1).otherwise(0)
    ).withColumn(
        "ooc_bucket_minus100_to_50",
        when(col("ooc_days").between(-100, 50), 1).otherwise(0)
    ).withColumn(
        "ooc_bucket_over_50",
        when(col("ooc_days") > 50, 1).otherwise(0)
    ).withColumn(
        "ooc_bucket_missing",
        when(col("ooc_days").isNull(), 1).otherwise(0)
    )
    
    # Drop original column
    result = result.drop("ooc_days")
    
    return result
