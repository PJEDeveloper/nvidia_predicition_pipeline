# pipeline/train_all_models.py

import os
import pandas as pd
import numpy as np
from datetime import datetime
from config.lookback_configs import LOOKBACK_MODEL_CONFIGS
from config.paths import STOCK_CSV_PATH, ENSEMBLE_INPUTS_DIR
from utils.preprocessing import prepare_features_and_sequences
from models.train_model import train_model
from utils.io_utils import ensure_dir
from joblib import dump


def train_all_models():
    # Load and preprocess base data
    df = pd.read_csv(STOCK_CSV_PATH, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for lookback, config in LOOKBACK_MODEL_CONFIGS.items():
        print(f"\nTraining model with lookback {lookback} days...")

        # Generate sequences
        X, y, dates, scaler = prepare_features_and_sequences(df.copy(), lookback)

        # Split chronologically
        split_1 = int(0.7 * len(X))
        split_2 = int(0.9 * len(X))
        X_train, y_train = X[:split_1], y[:split_1]
        X_val, y_val = X[split_1:split_2], y[split_1:split_2]
        X_test, y_test = X[split_2:], y[split_2:]
        test_dates = dates[split_2:]

        # Train model
        input_shape = (X_train.shape[1], X_train.shape[2])
        output_dir = os.path.join(ENSEMBLE_INPUTS_DIR, f"{lookback}D", f"run_{timestamp}")
        ensure_dir(output_dir)
        model, history = train_model(
            X_train, y_train, X_val, y_val,
            config=config,
            input_shape=input_shape,
            save_dir=output_dir,
            graph_base_name=f"lookback_{lookback}"
        )

        # Predict on all data
        y_pred_scaled = model.predict(X)
        y_pred = scaler['Close'].inverse_transform(y_pred_scaled)
        actual_close = scaler['Close'].inverse_transform(y.reshape(-1, 1))

        # Save prediction CSV
        output_df = pd.DataFrame({
            'Date': dates,
            'Actual_Close': actual_close.flatten(),
            'Predicted_Close': y_pred.flatten()
        })
        pred_path = os.path.join(output_dir, f"lookback_{lookback}_predictions.csv")
        output_df.to_csv(pred_path, index=False)
        print(f"Saved predictions to {pred_path}")


if __name__ == "__main__":
    train_all_models()