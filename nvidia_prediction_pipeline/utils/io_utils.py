# utils/io_utils.py

import os

def ensure_dir(path):
    """
    Creates the directory if it doesn't exist.
    Safe to call even if the directory already exists.
    """
    os.makedirs(path, exist_ok=True)


def save_dataframe(df, path, index=False):
    """
    Saves a pandas DataFrame to a CSV file after ensuring the directory exists.
    """
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=index)


def save_joblib(obj, path):
    """
    Saves a Python object to disk using joblib after ensuring the directory exists.
    """
    import joblib
    ensure_dir(os.path.dirname(path))
    joblib.dump(obj, path)


def load_joblib(path):
    """
    Loads a joblib file from disk.
    """
    import joblib
    return joblib.load(path)