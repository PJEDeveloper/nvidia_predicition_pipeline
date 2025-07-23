# models/train_model.py

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from models.model_builder import build_model
from utils.io_utils import ensure_dir


def train_model(X_train, y_train, X_val, y_val, config, input_shape, save_dir, graph_base_name="model"):
    """
    Trains a model using the given configuration and data.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        config (list): Model configuration
        input_shape (tuple): Shape of the input data
        save_dir (str): Directory to save model outputs
        graph_base_name (str): Prefix for saved plots

    Returns:
        model: Trained Keras model
        history: Training history object
    """
    ensure_dir(save_dir)

    model = build_model(config, input_shape)

    early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(save_dir, f"{graph_base_name}_loss.png")
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    plt.close()

    return model, history