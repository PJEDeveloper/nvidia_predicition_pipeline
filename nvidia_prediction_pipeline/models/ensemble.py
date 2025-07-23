# models/ensemble.py

import os
import glob
import re
import time
import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error

from config.paths import ENSEMBLE_INPUTS_DIR, META_MODEL_DIR, FEATURE_COLS_PATH, META_MODEL_PATH
from utils.io_utils import ensure_dir, save_joblib

def discover_latest_prediction_csvs(lookback_periods):
    csv_files = []

    for lookback in lookback_periods:
        lookback_path = os.path.join(ENSEMBLE_INPUTS_DIR, f"{lookback}D")
        subfolders = [d for d in glob.glob(os.path.join(lookback_path, "*")) if os.path.isdir(d)]
        if not subfolders:
            raise FileNotFoundError(f"No subfolders found in {lookback_path}")

        # Sort by datetime (assuming run_{timestamp})
        subfolders_sorted = sorted(
            subfolders,
            key=lambda x: time.strptime(x.split("_")[-2] + "_" + x.split("_")[-1], "%Y-%m-%d_%H-%M-%S"),
            reverse=True
        )

        latest_subfolder = subfolders_sorted[0]
        prediction_csvs = glob.glob(os.path.join(latest_subfolder, "*_predictions.csv"))
        if not prediction_csvs:
            raise FileNotFoundError(f"No *_predictions.csv found in {latest_subfolder}")

        csv_files.append(prediction_csvs[0])

    return csv_files

def train_meta_model(lookback_periods):
    csv_files = discover_latest_prediction_csvs(lookback_periods)

    dfs = []
    for f in csv_files:
        match = re.search(r"lookback_(\d+)_predictions\.csv", os.path.basename(f))
        if not match:
            raise ValueError(f"Could not parse lookback period from filename: {f}")
        lookback = match.group(1)
        df = pd.read_csv(f)
        df = df[["Date", "Predicted_Close"]]
        df = df.rename(columns={"Predicted_Close": f"Pred_{lookback}"})
        dfs.append(df)

    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on="Date", how="inner")

    # Actual close from any CSV
    actual_df = pd.read_csv(csv_files[0])[["Date", "Actual_Close"]]
    merged_df = pd.merge(merged_df, actual_df, on="Date", how="inner")

    merged_df["Date"] = pd.to_datetime(merged_df["Date"])

    print("Merged DataFrame shape:", merged_df.shape)
    print(merged_df.head())

    feature_cols = [c for c in merged_df.columns if c.startswith("Pred_")]
    X = merged_df[feature_cols].values
    y = merged_df["Actual_Close"].values

    ensure_dir(META_MODEL_DIR)
    model = Ridge(alpha=1.0)
    model.fit(X, y)

    # Evaluate
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    print(f"Meta-model R2: {r2:.4f}, MAE: {mae:.4f}")

    # Save model + feature order
    save_joblib(model, META_MODEL_PATH)
    save_joblib(feature_cols, FEATURE_COLS_PATH)
    print(f"Meta-model and feature cols saved to {META_MODEL_DIR}")

if __name__ == "__main__":
    LOOKBACKS = [365, 270, 180, 90, 60, 30, 14, 1]
    train_meta_model(LOOKBACKS)
