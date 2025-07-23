# utils/preprocessing.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def prepare_features_and_sequences(df, lookback):
    """
    Prepares engineered features and sequences for training.

    Args:
        df (pd.DataFrame): Raw historical stock data
        lookback (int): Lookback window in days

    Returns:
        X (np.ndarray): Feature sequences
        y (np.ndarray): Targets (Close price)
        dates (np.ndarray): Corresponding dates
        scalers (dict): Scalers used per feature
    """
    df = df.copy()

    # Feature engineering
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['STD50'] = df['Close'].rolling(window=50).std()
    df['Return1'] = df['Close'].pct_change()
    df['Return5'] = df['Close'].pct_change(5)
    df['Return20'] = df['Close'].pct_change(20)
    df['Bollinger_Upper'] = df['MA20'] + 2 * df['STD20']
    df['Bollinger_Lower'] = df['MA20'] - 2 * df['STD20']
    df['Range'] = df['High'] - df['Low']
    df['Close_Open'] = df['Close'] - df['Open']
    df['Lag1'] = df['Close'].shift(1)
    df['Rank20'] = df['Close'].rolling(20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])

    features = [
        'Close', 'Volume', 'Open', 'High', 'Low',
        'MA20', 'MA50', 'MA200',
        'STD20', 'STD50',
        'Return1', 'Return5', 'Return20',
        'Bollinger_Upper', 'Bollinger_Lower',
        'Range', 'Close_Open', 'Lag1', 'Rank20'
    ]

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Scale features
    scalers = {}
    scaled_data = []
    for col in features:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[[col]])
        scalers[col] = scaler
        scaled_data.append(scaled)

    scaled_array = np.hstack(scaled_data)

    # Create sequences
    X, y, dates = [], [], []
    for i in range(lookback, len(scaled_array)):
        X.append(scaled_array[i - lookback:i])
        y.append(scaled_array[i, 0])  # target is Close (first feature)
        dates.append(df['Date'].iloc[i])

    return np.array(X), np.array(y), np.array(dates), scalers