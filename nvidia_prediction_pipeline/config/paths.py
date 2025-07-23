# config/paths.py

import os

# Base directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
UTILS_DIR = os.path.join(BASE_DIR, "utils")
SENTIMENT_DIR = os.path.join(BASE_DIR, "sentiment")
PIPELINE_DIR = os.path.join(BASE_DIR, "pipeline")
ENSEMBLE_INPUTS_DIR = os.path.join(PIPELINE_DIR, "ensemble_inputs")
META_MODEL_DIR = os.path.join(PIPELINE_DIR, "meta_model")
NEWS_CACHE_DIR = os.path.join(SENTIMENT_DIR, "news_cache")

# Data files
STOCK_CSV_PATH = os.path.join(DATA_DIR, "nvidia_stock_data.csv")
FEATURE_COLS_PATH = os.path.join(META_MODEL_DIR, "feature_cols.joblib")
META_MODEL_PATH = os.path.join(META_MODEL_DIR, "meta_model_ridge.joblib")
PREDICTION_LOG_PATH = os.path.join(META_MODEL_DIR, "ensemble_prediction_log.csv")
NEWS_CACHE_PATH = os.path.join(NEWS_CACHE_DIR, "news_cache.json")

# API key
NEWS_API_KEY_PATH = os.path.join(BASE_DIR, "keys", "newsapi_key.txt")