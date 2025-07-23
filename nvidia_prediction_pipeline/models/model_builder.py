# models/model_builder.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GRU, Dense, Bidirectional
from models.attention import BahdanauAttention


def build_model(config, input_shape):
    """
    Builds a Sequential model based on the config and input shape.

    Args:
        config (list): List of (layer_type, units) tuples.
        input_shape (tuple): Shape of the input data (timesteps, features).

    Returns:
        keras.models.Sequential: Compiled model
    """
    model = Sequential()
    first_layer = True

    for layer_type, units in config:
        if layer_type == "Conv1D":
            if first_layer:
                model.add(Conv1D(units, kernel_size=3, activation="relu", input_shape=input_shape))
                first_layer = False
            else:
                model.add(Conv1D(units, kernel_size=3, activation="relu"))

        elif layer_type == "GRU":
            layer = GRU(units, return_sequences=True, dropout=0.2)
            if first_layer:
                model.add(Bidirectional(layer, input_shape=input_shape))
                first_layer = False
            else:
                model.add(Bidirectional(layer))

        elif layer_type == "Attention":
            model.add(BahdanauAttention(units))

        elif layer_type == "Dense":
            model.add(Dense(units))

        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae", "mse"]
    )

    return model