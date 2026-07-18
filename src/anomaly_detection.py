"""
Anomaly Detection Module

This module provides reusable functions for:

1. Feature Preparation
2. Feature Scaling
3. Z-Score Detection
4. Robust Z-Score Detection
5. Isolation Forest Detection
6. Local Outlier Factor Detection

Author:
BITS Pilani M.Tech Dissertation
"""

import numpy as np
import pandas as pd

from scipy.stats import zscore

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import IsolationForest

from sklearn.neighbors import LocalOutlierFactor

from src.feature_engineering import validate_required_columns


# =====================================================
# Feature Preparation
# =====================================================

def prepare_features(df):
    """
    Select anomaly detection features.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """

    features = [
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
        "thermal_stress"
    ]

    validate_required_columns(
        df,
        features
    )

    return df[features]


# =====================================================
# Feature Scaling
# =====================================================

def scale_features(X):
    """
    Standardize numerical features.

    Parameters
    ----------
    X : pandas.DataFrame

    Returns
    -------
    tuple
        scaler,
        X_scaled
    """

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return scaler, X_scaled


# =====================================================
# Z-Score Detection
# =====================================================

def detect_zscore(
    X_scaled,
    threshold=3
):
    """
    Detect anomalies using Z-Score.

    Parameters
    ----------
    X_scaled : ndarray

    threshold : float

    Returns
    -------
    ndarray
        Binary predictions

        1 = anomaly
        0 = normal
    """

    z_scores = np.abs(
        zscore(
            X_scaled,
            axis=0
        )
    )

    predictions = (
        z_scores > threshold
    ).any(axis=1).astype(int)

    return predictions


# =====================================================
# Robust Z-Score Detection
# =====================================================

def detect_robust_zscore(
    X_scaled,
    threshold=3.5
):
    """
    Detect anomalies using Robust Z-Score.

    Parameters
    ----------
    X_scaled : ndarray

    threshold : float

    Returns
    -------
    ndarray
    """

    median = np.median(
        X_scaled,
        axis=0
    )

    mad = np.median(
        np.abs(
            X_scaled - median
        ),
        axis=0
    )

    mad[mad == 0] = 1e-9

    robust_z = (
        0.6745
        *
        (
            X_scaled - median
        )
        / mad
    )

    predictions = (
        np.abs(
            robust_z
        ) > threshold
    ).any(axis=1).astype(int)

    return predictions

# =====================================================
# Isolation Forest
# =====================================================

def detect_isolation_forest(
    X_scaled,
    contamination=0.01,
    random_state=42
):
    """
    Detect anomalies using Isolation Forest.

    Parameters
    ----------
    X_scaled : ndarray

    contamination : float

    random_state : int

    Returns
    -------
    tuple
        (model, predictions)
    """

    model = IsolationForest(
        contamination=contamination,
        random_state=random_state
    )

    model.fit(X_scaled)

    predictions = model.predict(X_scaled)

    # Convert sklearn output
    # Normal  -> 0
    # Anomaly -> 1

    predictions = np.where(
        predictions == -1,
        1,
        0
    )

    return model, predictions


# =====================================================
# Local Outlier Factor
# =====================================================

def detect_lof(
    X_scaled,
    n_neighbors=20,
    contamination=0.01
):
    """
    Detect anomalies using Local Outlier Factor.

    Parameters
    ----------
    X_scaled : ndarray

    n_neighbors : int

    contamination : float

    Returns
    -------
    tuple
        (model, predictions)
    """

    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination
    )

    predictions = model.fit_predict(
        X_scaled
    )

    # Convert sklearn output
    # Normal  -> 0
    # Anomaly -> 1

    predictions = np.where(
        predictions == -1,
        1,
        0
    )

    return model, predictions

# =====================================================
# Run Complete Anomaly Detection Pipeline
# =====================================================

def run_all_anomaly_detection(df):
    """
    Run all anomaly detection methods.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    dict
        Contains scaled features, trained models,
        and predictions from all anomaly detectors.
    """

    # Prepare Features

    X = prepare_features(df)

    # Scale Features

    scaler, X_scaled = scale_features(X)

    # Z-Score

    z_predictions = detect_zscore(X_scaled)

    # Robust Z-Score

    robust_predictions = detect_robust_zscore(
        X_scaled
    )

    # Isolation Forest

    iso_model, iso_predictions = detect_isolation_forest(
        X_scaled
    )

    # Local Outlier Factor

    lof_model, lof_predictions = detect_lof(
        X_scaled
    )

    return {

        "features": X,

        "scaler": scaler,

        "X_scaled": X_scaled,

        "z_predictions": z_predictions,

        "robust_predictions": robust_predictions,

        "iso_model": iso_model,

        "iso_predictions": iso_predictions,

        "lof_model": lof_model,

        "lof_predictions": lof_predictions

    }