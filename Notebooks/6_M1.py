# Databricks notebook source
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt


# COMMAND ----------


gold_pd = spark.table("churn_gold.modeling_dataset").toPandas()

target = "churned"
drop_cols = ["unique_customer_identifier", "observation_date", "churned"]
features = [c for c in gold_pd.columns if c not in drop_cols]

X = gold_pd[features]
y = gold_pd[target]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Carve early stopping monitor from training data - keeps validation clean
X_train_fit, X_es, y_train_fit, y_es = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
    
print(f"Train: {X_train_fit.shape[0]}, ES monitor: {X_es.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
print(f"Churn rates - Train: {y_train_fit.mean():.3f}, Val: {y_val.mean():.3f}, Test: {y_test.mean():.3f}")

scale_pos_weight = (y_train_fit == 0).sum() / (y_train_fit == 1).sum()

model = XGBClassifier(
    objective="binary:logistic", n_estimators=1000, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8,
    min_child_weight=5, reg_alpha=0.1, scale_pos_weight=scale_pos_weight, eval_metric="auc", early_stopping_rounds=50,
    random_state=42, n_jobs=-1)  
model.fit(X_train_fit, y_train_fit, eval_set=[(X_train_fit, y_train_fit), (X_es, y_es)], verbose=50)    

print(f"\nBest iteration: {model.best_iteration}")

# Plot early stopping curve
results = model.evals_result()
train_auc = results["validation_0"]["auc"]
es_auc = results["validation_1"]["auc"]

plt.figure(figsize=(10, 5))
plt.plot(train_auc, label="Train AUC", color="steelblue")
plt.plot(es_auc, label="Early Stop Monitor AUC", color="darkorange")
plt.axvline(x=model.best_iteration, color="red", linestyle="--", label=f"Best iteration: {model.best_iteration}")
plt.xlabel("Number of Trees")
plt.ylabel("AUC")
plt.title("XGBoost Early Stopping Curve")
plt.legend()
plt.tight_layout()
plt.show()

# 5-fold CV on training data using best iteration
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_model = XGBClassifier(
    objective="binary:logistic",
    n_estimators=model.best_iteration,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

cv_scores = cross_val_score(cv_model, X_train, y_train, cv=outer_cv, scoring="roc_auc")
print(f"\n5-Fold CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Validation
val_preds = model.predict_proba(X_val)[:, 1]
print(f"Validation AUC: {roc_auc_score(y_val, val_preds):.4f}")
print(classification_report(y_val, model.predict(X_val)))

# Test
test_preds = model.predict_proba(X_test)[:, 1]
print(f"Test AUC: {roc_auc_score(y_test, test_preds):.4f}")
print(classification_report(y_test, model.predict(X_test)))

# COMMAND ----------

print(f"Rows: {gold_pd.shape[0]:,}")
print(f"Features: {len(features)}")
print(f"Train / Val / Test: {X_train_fit.shape[0]:,} / {X_val.shape[0]:,} / {X_test.shape[0]:,}")
print(f"Best iteration (trees): {model.best_iteration}")
print(f"5-Fold CV AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
print(f"Validation AUC: {roc_auc_score(y_val, val_preds):.4f}")
print(f"Test AUC: {roc_auc_score(y_test, test_preds):.4f}")
print(f"Learning rate: {model.learning_rate}")
print(f"Max depth: {model.max_depth}")
print(f"Subsample: {model.subsample}")
print(f"Colsample bytree: {model.colsample_bytree}")
print(f"Min child weight: {model.min_child_weight}")
print(f"Reg alpha (L1): {model.reg_alpha}")
print(f"Scale pos weight: {scale_pos_weight:.3f}")

# COMMAND ----------

print(gold_pd.columns.tolist())

# COMMAND ----------

importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

plt.figure(figsize=(10, 8))
importance.head(20).plot(kind="barh")
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance (Gain)")
plt.title("Top 20 Most Predictive Features")
plt.tight_layout()
plt.show()

importance.head(20).to_string()

# COMMAND ----------

test_preds = model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, test_preds)
print(f"Test AUC: {test_auc:.4f}")
print(classification_report(y_test, model.predict(X_test)))

# COMMAND ----------


