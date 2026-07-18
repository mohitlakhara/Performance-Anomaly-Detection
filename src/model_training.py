"""
Model Training Module

This module provides reusable functions for:

1. Train-Test Split
2. Random Forest Training
3. XGBoost Training
4. Model Saving

Author:
BITS Pilani M.Tech Dissertation
"""

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.feature_engineering import validate_required_columns

from src.config import MODELS_DIR

def split_dataset(df):
    """
    Split dataset into training and testing sets.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
    """

    validate_required_columns(
        df,
        [
            "cpu_utilization",
            "memory_usage",
            "disk_io",
            "network_latency",
            "process_count",
            "thread_count",
            "context_switches",
            "cache_miss_rate",
            "temperature",
            "power_consumption",
            "uptime",
            "resource_stress_index",
            "cpu_memory_ratio",
            "disk_pressure",
            "context_switch_density",
            "cache_efficiency",
            "thermal_stress",
            "risk_label"
        ]
    )

    features = [
    'cpu_utilization',
    'memory_usage',
    'disk_io',
    'network_latency',
    'process_count',
    'thread_count',
    'context_switches',
    'cache_miss_rate',
    'temperature',
    'power_consumption',
    'uptime',
    'resource_stress_index',
    'cpu_memory_ratio',
    'disk_pressure',
    'context_switch_density',
    'cache_efficiency',
    'thermal_stress'
    ]

    X = df[features]

    y = df["risk_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

# =====================================================
# Train Random Forest
# =====================================================

def train_random_forest(X_train, y_train):
    """
    Train Random Forest classifier.

    Parameters
    ----------
    X_train : pandas.DataFrame
    y_train : pandas.Series

    Returns
    -------
    RandomForestClassifier
    """

    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    return rf_model

# =====================================================
# Train XGBoost
# =====================================================

def train_xgboost(X_train, y_train):
    """
    Train XGBoost classifier.

    Parameters
    ----------
    X_train : pandas.DataFrame
    y_train : pandas.Series

    Returns
    -------
    XGBClassifier
    """

    xgb_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        eval_metric="mlogloss"
    )

    xgb_model.fit(X_train, y_train)

    return xgb_model

# =====================================================
# Save Model
# =====================================================

def save_model(model, filename):
    """
    Save trained model.

    Parameters
    ----------
    model : object

    filename : str
    """

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODELS_DIR / filename
    )

    print(f"Model Saved: {filename}")

# =====================================================
# Train All Models
# =====================================================

def train_all_models(df):
    """
    Train all machine learning models.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    dict
    """

    X_train, X_test, y_train, y_test = split_dataset(df)

    rf_model = train_random_forest(
        X_train,
        y_train
    )

    xgb_model = train_xgboost(
        X_train,
        y_train
    )

    save_model(
        rf_model,
        "random_forest_model.pkl"
    )

    save_model(
        xgb_model,
        "xgboost_model.pkl"
    )

    return {

        "Random Forest": rf_model,

        "XGBoost": xgb_model,

        "X_train": X_train,

        "X_test": X_test,

        "y_train": y_train,

        "y_test": y_test

    }
