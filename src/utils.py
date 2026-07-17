"""
Common Utility Functions
"""

from pathlib import Path
import pandas as pd


def create_directory(path):
    """
    Create directory if it does not exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def save_dataframe(df, filepath):
    """
    Save DataFrame to CSV.
    """
    create_directory(Path(filepath).parent)
    df.to_csv(filepath, index=False)


def print_section(title):
    """
    Print formatted section header.
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)