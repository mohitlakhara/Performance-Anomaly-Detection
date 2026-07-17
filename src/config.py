"""
Project Configuration
Performance Anomaly Detection
BITS Pilani M.Tech Dissertation
"""

from pathlib import Path

# =====================================================
# PROJECT PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

HDFS_DATA_DIR = DATA_DIR / "hdfs"

NAB_DATA_DIR = DATA_DIR / "NAB"

MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR = PROJECT_ROOT / "reports"

FIGURES_DIR = REPORTS_DIR / "figures"

RESULTS_DIR = REPORTS_DIR / "results"

# =====================================================
# RANDOM STATE
# =====================================================

RANDOM_STATE = 42

# =====================================================
# DATA SPLIT
# =====================================================

TEST_SIZE = 0.20

# =====================================================
# ISOLATION FOREST
# =====================================================

ISO_CONTAMINATION = 0.01

HDFS_CONTAMINATION = 0.029

# =====================================================
# LOF
# =====================================================

LOF_NEIGHBORS = 50

# =====================================================
# RANDOM FOREST
# =====================================================

RF_ESTIMATORS = 200

RF_MAX_DEPTH = 10

# =====================================================
# XGBOOST
# =====================================================

XGB_ESTIMATORS = 200

XGB_MAX_DEPTH = 6

XGB_LEARNING_RATE = 0.1