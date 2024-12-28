import pandas as pd 
import polars as pl 
import numpy as np
import os
from glob import glob

# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report,  roc_curve,auc, roc_auc_score, r2_score

from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tqdm import tqdm

import numpy as np
from sklearn.metrics import r2_score

def time_series_error_metrics(y_true, y_pred, weights):
    """
        - Mean Absolute Error (MAE)
        - Mean Squared Error (MSE)
        - Root Mean Squared Error (RMSE)
        - Mean Absolute Percentage Error (MAPE)
        - Symmetric Mean Absolute Percentage Error (sMAPE)
        - Mean Directional Accuracy (MDA)
        - R-squared (R2)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)    
    weights = np.array(weights) / np.sum(weights)  # Normalize weights to sum to 1
    
    # Compute weighted errors
    mae = np.sum(weights * np.abs(y_true - y_pred))
    mse = np.sum(weights * (y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    # Weighted MAPE (avoid division by zero by handling small values of y_true)
    mape = np.sum(weights * np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    # Weighted sMAPE (avoid division by zero by handling small values in the denominator)
    smape = np.sum(weights * (2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-10))) * 100

    # Mean Directional Accuracy (MDA)
    directional_accuracy = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred)))
    mda = directional_accuracy * 100

    # Weighted R-squared
    r2 = r2_score(y_true, y_pred, sample_weight=weights)
    
    # Return results as a dictionary
    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "sMAPE": smape,
        "MDA (%)": mda,
        "R2": r2,
    }
    
    return metrics


# train_df=pd.read_parquet('./data/all_data.parquet')
# print(f"all data shape: {train_df.shape}")
# print(f"all data columns: {train_df.columns}")
# print(f"all data head: {train_df.head()}")

X_train=pd.read_parquet('./data/X_train.parquet')
X_test=pd.read_parquet('./data/X_test.parquet')
y_train=pd.read_parquet('./data/y_train.parquet').squeeze()
y_test=pd.read_parquet('./data/y_test.parquet').squeeze()
weights_train=pd.read_parquet('./data/weights_train.parquet').squeeze()
weights_test=pd.read_parquet('./data/weights_test.parquet').squeeze()
print(type(X_train), type(X_test), type(y_train), type(y_test), type(weights_train), type(weights_test))
print(f'X_train{X_train.shape}, X_test{X_test.shape},\ny_train{y_train.shape}, y_test{y_test.shape}, \nweights_train{weights_train.shape}, weights_test{weights_test.shape}')
print(f'X_train column:\n {X_train.columns}, X_test column:\n {X_test.columns}, y_train column:\n {y_train.name}, y_test column:\n {y_test.name}, weights_train column:\n {weights_train.name}, weights_test column:\n {weights_test.name}')
print(f'X_train head:\n {X_train.head()}, X_test head:\n {X_test.head()}, y_train head:\n {y_train.head()}, y_test head:\n {y_test.head()}, weights_train head:\n {weights_train.head()}, weights_test head:\n {weights_test.head()}')

model=XGBRegressor()
model.fit(X_train, y_train, sample_weight=weights_train)
# predictions on the test data
y_pred = model.predict(X_test)
metrics=time_series_error_metrics(y_true=y_test, y_pred=y_pred, weights = weights_test)
print(metrics)


## mutual information
from sklearn.feature_selection import mutual_info_regression


mi_scores = mutual_info_regression(X_train, y_train, random_state=42)
mi_scores = pd.Series(mi_scores, index=X_train.columns).sort_values(ascending=False)


top_features = mi_scores.head(10).index
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]


model = XGBRegressor()
model.fit(X_train_selected, y_train, sample_weight=weights_train)


y_pred = model.predict(X_test_selected)
metrics = time_series_error_metrics(y_true=y_test, y_pred=y_pred, weights=weights_test)
print("Metrics (Mutual Information):", metrics)


## PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=10, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


model = XGBRegressor()
model.fit(X_train_pca, y_train, sample_weight=weights_train)

y_pred = model.predict(X_test_pca)
metrics = time_series_error_metrics(y_true=y_test, y_pred=y_pred, weights=weights_test)
print("Metrics (PCA):", metrics)


## RF
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train, sample_weight=weights_train)
feature_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

top_features = feature_importances.head(10).index
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]


model = XGBRegressor()
model.fit(X_train_selected, y_train, sample_weight=weights_train)

y_pred = model.predict(X_test_selected)
metrics = time_series_error_metrics(y_true=y_test, y_pred=y_pred, weights=weights_test)
print("Metrics (Random Forest Feature Importance):", metrics)
