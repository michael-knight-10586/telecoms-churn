# For a brief overview please see PPT 

# Telecoms Customer Churn Prediction

End-to-end churn prediction pipeline built for UK Telecoms LTD using Azure Databricks, PySpark, and XGBoost. Demonstrates production-level MLOps practices including medallion architecture, modular feature engineering.

## Problem Statement

Predict which broadband customers are likely to place a cease request, enabling the retention team to prioritise outreach to high-risk customers before they leave.

## Architecture

The pipeline follows a medallion architecture across Delta Lake:

- **Bronze** - raw data landed as-is from source files (cease, customer_info, calls, usage)
- **Silver** - cleaned and joined dataset with one row per customer at their observation point
- **Gold** - engineered features ready for modelling

## Notebooks

Run in order:

1. `create_raw_info` - builds the silver observation dataset joining cease to customer info
2. `create_features_1_info` - engineers contract status, OOC buckets, technology, speed, sales channel, CRM package, tenure features
3. `create_features_2_calls` - engineers call behaviour features (has_calls binary flag)
4. `create_features_3_usage_gold` - engineers usage bucket features and joins all feature sets into the gold modelling dataset

## Source Modules

- `src/dataset_creation.py` - silver dataset construction functions
- `src/features_customer_info.py` - customer info feature engineering functions
- `src/features_calls.py` - calls feature engineering functions
- `src/features_usage.py` - usage feature engineering functions

## Model

XGBoost binary classifier with the following setup:

- 60/20/20 train/validation/test split (stratified)
- Early stopping on a carved-out 10% ES monitor set
- 5-fold cross validation on training data for generalisation estimate
- L1 regularisation (reg_alpha = 0.1)
- Class imbalance handled via scale_pos_weight

**Results: 0.937 AUC (5-fold CV), 0.937 validation, 0.936 test**

## Data

Synthetic UK telecoms data (~183k customers, 61% churn rate). Four source files: cease.csv, customer_info.parquet, calls.csv, usage.parquet.

## Setup

1. Clone repo and connect to Databricks via Repos
2. Upload source data files to DBFS
3. Run notebooks in sequence
4. Cluster: DBR 13.3 LTS ML Runtime or higher

# \- \*\*ML Framework\*\*: MLflow, scikit-learn, XGBoost

# \- \*\*Orchestration\*\*: Databricks Jobs

# \- \*\*Version Control\*\*: GitHub

# 

# \## Data Sources

# 1\. \*\*Cease Data\*\*: Customer cancellation records

# 2\. \*\*Customer Info\*\*: Contract status, tenure, service details

# 3\. \*\*Call Data\*\*: Customer service interaction history

# 4\. \*\*Usage Data\*\*: Broadband consumption patterns

# 

# \## Pipeline Architecture

# 

# \### Bronze Layer (Raw Data Ingestion)

# \- Ingest raw CSV and Parquet files

# \- Store as Delta tables with minimal transformation

# \- Preserve data lineage

# 

# \### Silver Layer (Cleansed \& Conformed)

# \- Data quality checks and validation

# \- Standardization and type casting

# \- Join customer dimensions

# 

# \### Gold Layer (Feature Engineering)

# \- Aggregate features from usage and call patterns

# \- Calculate customer behavior metrics

# \- Create model-ready feature set

# 

# \### Model Training \& Deployment

# \- MLflow experiment tracking

# \- Model versioning in MLflow Registry

# \- Performance evaluation and comparison

# 

# \## Setup Instructions

# 1\. Clone this repository

# 2\. Connect to Databricks workspace via Databricks Repos

# 3\. Create cluster with DBR ML Runtime 13.3 LTS or higher

# 4\. Run notebooks in sequence from 01 through 05

# 

# \## Project Structure

# ```

# ├── notebooks/         # Databricks notebooks by pipeline stage

# ├── src/               # Reusable Python modules

# ├── config/            # Configuration files

# ├── tests/             # Unit and integration tests

# ├── docs/              # Additional documentation

# └── requirements.txt   # Python dependencies

# ```

# 

# 






