"""
Data Preprocessing Module

This module provides reusable preprocessing functions for:
1. IT System Metrics Dataset
2. HDFS Dataset
3. NAB Dataset

Author:
BITS Pilani M.Tech Dissertation
"""

from pathlib import Path

import pandas as pd
import numpy as np

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)

from src.utils import (
    create_directory,
    save_dataframe,
    print_section,
)


# =====================================================
# Generic Dataset Loader
# =====================================================

def load_dataset(file_path):
    """
    Load a CSV dataset.

    Parameters
    ----------
    file_path : Path

    Returns
    -------
    pandas.DataFrame
    """

    print_section("Loading Dataset")

    df = pd.read_csv(file_path)

    print(f"Dataset Loaded Successfully")

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    return df


# =====================================================
# Dataset Summary
# =====================================================

def dataset_summary(df):
    """
    Print dataset summary.
    """

    print_section("Dataset Summary")

    print(df.info())

    print("\nMissing Values")

    print(df.isnull().sum())

    print("\nDuplicate Rows")

    print(df.duplicated().sum())


# =====================================================
# Remove Duplicates
# =====================================================

def remove_duplicates(df):

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    print()

    print(f"Duplicates Removed : {before-after}")

    return df


# =====================================================
# Validate Numeric Columns
# =====================================================

def validate_numeric_columns(df):

    numeric_columns = df.select_dtypes(include=np.number).columns

    print_section("Numeric Column Validation")

    for column in numeric_columns:

        print(
            f"{column:<30}"
            f" Min={df[column].min():.4f}"
            f" Max={df[column].max():.4f}"
        )


# =====================================================
# Save Clean Dataset
# =====================================================

def save_clean_dataset(df, filename):

    create_directory(PROCESSED_DATA_DIR)

    filepath = PROCESSED_DATA_DIR / filename

    save_dataframe(df, filepath)

    print()

    print(f"Saved : {filepath}")


# =====================================================
# IT Dataset Pipeline
# =====================================================

def preprocess_it_dataset(filename):

    filepath = RAW_DATA_DIR / filename

    df = load_dataset(filepath)

    dataset_summary(df)

    df = remove_duplicates(df)

    validate_numeric_columns(df)

    save_clean_dataset(df, filename)

    return df