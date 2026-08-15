"""
Dataset creation functions for churn prediction model
"""

def create_live_customer_baseline(customer_info_df, cease_df, buffer_months=3):

    from pyspark.sql.functions import (
    lit, col, rand, row_number, add_months, max as spark_max, trunc
    )
    from pyspark.sql.window import Window
    """
    Create baseline dataset for live (non-churned) customers.
    
    Args:
        customer_info_df: Customer info DataFrame with monthly snapshots
        cease_df: Cease DataFrame with churn events
        buffer_months: Months between behavior window and end of data (default 3)
        
    Returns:
        DataFrame with unique_customer_identifier, churned (0), behav_win, 
        plus all customer_info columns from sampled month
    """
    # Identify customers who never churned
    unique_customers = customer_info_df.select("unique_customer_identifier").distinct()
    churned_customers = cease_df.select("unique_customer_identifier").distinct()
    live_customers = unique_customers.join(
        churned_customers, 
        on="unique_customer_identifier", 
        how="left_anti"
    )
    
    # Get max date and validate live customers have data through end
    max_date_in_data = customer_info_df.select(spark_max("datevalue")).collect()[0][0]
    last_customer_month = customer_info_df.groupBy("unique_customer_identifier").agg(
        spark_max("datevalue").alias("last_customer_month")
    )
    
    live_customers_complete = live_customers.join(
        last_customer_month, 
        on="unique_customer_identifier", 
        how="inner"
    ).filter(col("last_customer_month") == lit(max_date_in_data))
    
    # Sample random month with buffer
    cutoff_date = add_months(lit(max_date_in_data), -buffer_months)
    eligible_months = customer_info_df.join(
        live_customers_complete.select("unique_customer_identifier"),
        on="unique_customer_identifier",
        how="inner"
    ).select("unique_customer_identifier", "datevalue").distinct() \
     .filter(col("datevalue") <= cutoff_date)
    
    window_spec = Window.partitionBy("unique_customer_identifier").orderBy(rand())
    sampled_months = eligible_months.withColumn("rn", row_number().over(window_spec)) \
        .filter(col("rn") == 1) \
        .select("unique_customer_identifier", col("datevalue").alias("behav_win"))
    
    # Join customer_info data from sampled month
    live_baseline = sampled_months.join(
        customer_info_df,
        (sampled_months.unique_customer_identifier == customer_info_df.unique_customer_identifier) &
        (sampled_months.behav_win == customer_info_df.datevalue),
        how="inner"
    ).drop(customer_info_df.unique_customer_identifier).drop(customer_info_df.datevalue)

    # Add churned flag
    live_baseline = live_baseline.withColumn("churned", lit(0))

    return live_baseline

def create_churned_customer_baseline(customer_info_df, cease_df):
    from pyspark.sql.functions import trunc, col, add_months, lit
    
    """
    Create baseline dataset for churned customers with churn reason.
    
    Args:
        customer_info_df: Customer info DataFrame with monthly snapshots
        cease_df: Cease DataFrame with churn events
        
    Returns:
        DataFrame with unique_customer_identifier, churned (1), behav_win,
        reason_description_insight, plus all customer_info columns
    """
    # Get unique churned customers with their cease date
    churned_customers = cease_df.select(
        "unique_customer_identifier", 
        "cease_placed_date",
        "reason_description_insight"
    ).distinct()
    
    # Calculate behav_win and cease_month
    churned_with_behav_win = churned_customers.withColumn(
        "cease_month",
        trunc(col("cease_placed_date"), "month")
    ).withColumn(
        "behav_win",
        add_months(col("cease_month"), -1)
    )
    
    # Join to get one reason per customer-month (handles duplicates)
    from pyspark.sql.functions import first
    churned_with_reason = churned_with_behav_win.groupBy(
        "unique_customer_identifier", "behav_win", "cease_month"
    ).agg(
        first("reason_description_insight").alias("reason_description_insight")
    )
    
    # Join customer_info data from that month
    churned_baseline = churned_with_reason.alias("churn_temp").join(
        customer_info_df.alias("info"),
        (col("churn_temp.unique_customer_identifier") == col("info.unique_customer_identifier")) &
        (col("churn_temp.behav_win") == col("info.datevalue")),
        how="inner"
    ).select(
        col("churn_temp.unique_customer_identifier"),
        col("churn_temp.behav_win"),
        lit(1).alias("churned"),
        col("churn_temp.reason_description_insight"),
        *[col(f"info.{c}") for c in customer_info_df.columns if c not in ['unique_customer_identifier', 'datevalue']]
    )
    
    return churned_baseline

def filter_churned_customers(raw_info_df, cease_df, reasons_to_exclude=['Bereavement', 'HomeMove', 'BadDebtDisconnect']):
    """
    Join cease data to raw info and filter out specific churn reasons.
    """
    from pyspark.sql.functions import row_number
    from pyspark.sql.window import Window
    
    # Join cease data to raw info (left join to keep all raw_info rows)
    joined_df = raw_info_df.join(
        cease_df.select("unique_customer_identifier", "reason_description_insight"),
        on="unique_customer_identifier",
        how="left"
    )
    
    # Filter: Keep non-churned OR churned with reasons NOT in exclusion list
    filtered_df = joined_df.filter(
        (col("churned") == 0) |  # Keep all non-churned
        ((col("churned") == 1) & (col("reason_description_insight").isNotNull()) & 
         (~col("reason_description_insight").isin(reasons_to_exclude)))
    )
    
    # Drop the reason column
    filtered_df = filtered_df.drop("reason_description_insight")
    
    # Remove duplicates
    window_spec = Window.partitionBy("unique_customer_identifier").orderBy("behav_win")
    filtered_df = filtered_df.withColumn("rn", row_number().over(window_spec)) \
        .filter(col("rn") == 1) \
        .drop("rn")
    
    return filtered_df
