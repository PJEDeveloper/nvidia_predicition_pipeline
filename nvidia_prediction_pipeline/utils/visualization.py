# utils/visualization.py

import os
import matplotlib.pyplot as plt
from utils.io_utils import ensure_dir


def plot_loss(history, save_path=None, show=False):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path)
        plt.close()
    elif show:
        plt.show()


def plot_predictions(dates, actual, predicted, title='Predicted vs Actual Close Price', save_path=None, show=False):
    plt.figure(figsize=(14, 6))
    plt.plot(dates, actual, label='Actual Close Price')
    plt.plot(dates, predicted, label='Predicted Close Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path)
        plt.close()
    elif show:
        plt.show()
