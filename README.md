# \# Customer Churn Prediction - MLOps Pipeline

# 

# \## Project Overview

# Predictive model to identify customers at risk of churning for UK Telecoms, enabling proactive retention efforts.

# 

# \## Business Problem

# Prioritize retention resources by identifying customers most likely to leave, improving retention team efficiency and reducing customer churn.

# 

# \## Technical Stack

# \- \*\*Platform\*\*: Azure Databricks

# \- \*\*Storage\*\*: Delta Lake (Medallion Architecture)

# \- \*\*Processing\*\*: PySpark

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

