"""
Feature Engineering Module

This module provides reusable feature engineering functions for:
1. IT System Metrics Dataset
2. Resource Stress Indicators
3. Derived Performance Features

Author:
BITS Pilani M.Tech Dissertation
"""

import pandas as pd
import numpy as np

# =====================================================
# Validate Required Columns
# =====================================================

def validate_required_columns(df, required_columns):
    """
    Validate whether required columns exist.

    Parameters
    ----------
    df : pandas.DataFrame

    required_columns : list

    Raises
    ------
    ValueError
    """

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )
    
# =====================================================
# CPU Memory Ratio
# =====================================================

def create_cpu_memory_ratio(df):
    """
    Create CPU Memory Ratio.

    Formula
    -------
    cpu_utilization / (memory_usage + 1)

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
            "cpu_utilization",
            "memory_usage"
        ]
    )

    df["cpu_memory_ratio"] = (
        df["cpu_utilization"]
        /
        (df["memory_usage"] + 1)
    )

    return df

# =====================================================
# Disk Pressure
# =====================================================

def create_disk_pressure(df):
    """
    Create Disk Pressure Feature.
    """

    validate_required_columns(
        df,
        [
            "disk_io",
            "process_count"
        ]
    )

    df["disk_pressure"] = (
        df["disk_io"]
        /
        (df["process_count"] + 1)
    )

    return df

# =====================================================
# Context Switch Density
# =====================================================

def create_context_switch_density(df):
    """
    Create Context Switch Density.
    """

    validate_required_columns(
        df,
        [
            "context_switches",
            "thread_count"
        ]
    )

    df["context_switch_density"] = (
        df["context_switches"]
        /
        (df["thread_count"] + 1)
    )

    return df

# =====================================================
# Cache Efficiency
# =====================================================

def create_cache_efficiency(df):
    """
    Create Cache Efficiency Feature.
    """

    validate_required_columns(
        df,
        [
            "cache_miss_rate"
        ]
    )

    df["cache_efficiency"] = (
        100
        -
        df["cache_miss_rate"]
    )

    return df

# =====================================================
# Thermal Stress
# =====================================================

def create_thermal_stress(df):
    """
    Create Thermal Stress Feature.
    """

    validate_required_columns(
        df,
        [
            "temperature",
            "power_consumption"
        ]
    )

    df["thermal_stress"] = (
        (
            df["temperature"]
            *
            df["power_consumption"]
        )
        / 100
    )

    return df

# =====================================================
# Resource Stress Index
# =====================================================

def create_resource_stress_index(df):
    """
    Create Resource Stress Index.
    """

    validate_required_columns(
        df,
        [
            "cpu_utilization",
            "memory_usage",
            "network_latency"
        ]
    )

    # For example, if your notebook uses:
    #
    # 0.4 * CPU + 0.4 * Memory + 0.2 * Latency
    #
    # preserve those exact weights.

    df["resource_stress_index"] = (
        0.4 * df["cpu_utilization"] +
        0.4 * df["memory_usage"] +
        0.2 * df["network_latency"]
    )

    return df

# =====================================================
# Create All Features
# =====================================================

def create_all_features(df):
    """
    Create all engineered features.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """

    df = create_resource_stress_index(df)
    df = create_cpu_memory_ratio(df)
    df = create_disk_pressure(df)
    df = create_context_switch_density(df)
    df = create_cache_efficiency(df)
    df = create_thermal_stress(df)

    return df

