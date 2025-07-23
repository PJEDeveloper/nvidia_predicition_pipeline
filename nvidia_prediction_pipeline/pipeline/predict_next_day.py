# pipeline/predict_next_day.py

import os
import glob
import pandas as pd
from datetime import datetime
from config.paths import (
    ENSEMBLE_INPUTS_DIR, FEATURE_COLS_PATH, META_MODEL_PATH,
    NEWS_CACHE_PATH, PREDICTION_LOG_PATH
)
from sentiment.news_fetcher import get_recent_news_with_cache
from sentiment.sentiment_analyzer import compute_sentiment_scores
from sentiment.sentiment_confidence import classify_confidence
from utils.io_utils import load_joblib, ensure_dir
from sklearn.linear_model import Ridge


LOOKBACK_PERIODS = [365, 270, 180, 90, 60, 30, 14, 1]


def get_latest_predictions():
    latest_preds = {}
    actual_close = None

    for lookback in LOOKBACK_PERIODS:
        folder = os.path.join(ENSEMBLE_INPUTS_DIR, f"{lookback}D")
        subfolders = sorted(glob.glob(os.path.join(folder, "*")), reverse=True)
        latest_folder = subfolders[0]
        pred_files = glob.glob(os.path.join(latest_folder, "*_predictions.csv"))
        df = pd.read_csv(pred_files[0])
        last_row = df.iloc[-1]
        latest_preds[f"Pred_{lookback}"] = last_row["Predicted_Close"]
        actual_close = last_row["Actual_Close"]

    return latest_preds, actual_close


def predict():
    # Load latest predictions and meta model
    latest_preds, actual_close = get_latest_predictions()
    X_input = pd.DataFrame([latest_preds])

    feature_cols = load_joblib(FEATURE_COLS_PATH)
    X_input = X_input.reindex(columns=feature_cols)

    model: Ridge = load_joblib(META_MODEL_PATH)
    ensemble_pred = model.predict(X_input)[0]

    # Sentiment analysis
    articles = get_recent_news_with_cache(NEWS_CACHE_PATH)
    avg_scores = compute_sentiment_scores(articles)
    confidence = classify_confidence(avg_scores)

    # Output
    delta_pct = (ensemble_pred - actual_close) / actual_close * 100
    print("\n--------------------------------------------")
    print(f"Predicted Close for Next Day: ${ensemble_pred:.2f}")
    print(f"Today's Close: ${actual_close:.2f}")
    print(f"Predicted Change: {delta_pct:.2f}%")
    print(f"Confidence: {confidence}")
    print(f"Sentiment: Pos={avg_scores['positive']:.2f}, Neu={avg_scores['neutral']:.2f}, Neg={avg_scores['negative']:.2f}")
    print("--------------------------------------------")

    # Logging
    now = datetime.now()
    log_data = {
        "Date_Predicted": [now.strftime("%Y-%m-%d")],
        "Timestamp": [now.strftime("%Y-%m-%d %H:%M:%S")],
        "Ensemble_Predicted_Close": [ensemble_pred],
        "Last_Close": [actual_close],
        "Predicted_Change_Percent": [delta_pct],
        "Confidence": [confidence],
        "Avg_Positive": [avg_scores['positive']],
        "Avg_Neutral": [avg_scores['neutral']],
        "Avg_Negative": [avg_scores['negative']],
    }
    log_data.update({k: [v] for k, v in latest_preds.items()})

    log_df = pd.DataFrame(log_data)
    ensure_dir(os.path.dirname(PREDICTION_LOG_PATH))

    if os.path.exists(PREDICTION_LOG_PATH):
        headers = pd.read_csv(PREDICTION_LOG_PATH, nrows=0).columns.tolist()
        log_df = log_df[headers]
        log_df.to_csv(PREDICTION_LOG_PATH, mode="a", header=False, index=False)
    else:
        log_df.to_csv(PREDICTION_LOG_PATH, index=False)

    print(f"\nPrediction logged to {PREDICTION_LOG_PATH}")


if __name__ == "__main__":
    predict()