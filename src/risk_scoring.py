"""
Risk Scoring Module

This module provides reusable functions for:

1. Risk Score Calculation
2. Risk Level Assignment
3. Risk Label Generation

Author:
BITS Pilani M.Tech Dissertation
"""

import pandas as pd
import numpy as np

from src.feature_engineering import validate_required_columns

# =====================================================
# Calculate Risk Score
# =====================================================

def calculate_risk_score(df):
    """
    Calculate composite system risk score.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """
    validate_required_columns(
    df,
    [
        "resource_stress_index",
        "cpu_memory_ratio",
        "disk_pressure",
        "context_switch_density",
        "cache_efficiency",
        "thermal_stress"
    ]
    )

    df["risk_score"] = (
        0.35*(df["cpu_utilization"]/100)
        +
        0.30*(df["memory_usage"]/100)
        +
        0.20*(df["network_latency"]/200)
        +
        0.15*(df["disk_io"]/100)
    )
    return df

# =====================================================
# Assign Risk Level
# =====================================================

def assign_risk_level(df):
    def get_level(score):

        if score < 30:
            return "Low"

        elif score < 60:
            return "Medium"

        return "High"

    df["risk_level"] = df["risk_score"].apply(get_level)
    return df

def assign_risk_label(df):

    mapping = {

        "Low":0,

        "Medium":1,

        "High":2

    }

    df["risk_label"] = df["risk_level"].map(mapping)

    return df

def create_risk_features(df):

    df = calculate_risk_score(df)

    df = assign_risk_level(df)

    df = assign_risk_label(df)

    return df

