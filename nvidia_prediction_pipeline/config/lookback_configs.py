# config/lookback_configs.py

LOOKBACK_MODEL_CONFIGS = {
    365: [
        ("Conv1D", 64),
        ("GRU", 250),
        ("GRU", 250),
        ("GRU", 250),
        ("GRU", 250),
        ("Attention", 128),
        ("Dense", 1)
    ],
    270: [
        ("Conv1D", 64),
        ("GRU", 250),
        ("GRU", 250),
        ("GRU", 250),
        ("Attention", 128),
        ("Dense", 1)
    ],
    180: [
        ("Conv1D", 64),
        ("GRU", 250),
        ("GRU", 250),
        ("GRU", 250),
        ("GRU", 150),
        ("Attention", 128),
        ("Dense", 1)
    ],
    90: [
        ("Conv1D", 64),
        ("GRU", 250),
        ("GRU", 250),
        ("Attention", 128),
        ("Dense", 1)
    ],
    60: [
        ("Conv1D", 64),
        ("GRU", 250),
        ("GRU", 250),
        ("Attention", 128),
        ("Dense", 1)
    ],
    30: [
        ("Conv1D", 64),
        ("GRU", 250),
        ("GRU", 250),
        ("Attention", 128),
        ("Dense", 1)
    ],
    14: [
        ("Conv1D", 64),
        ("GRU", 250),
        ("GRU", 250),
        ("Attention", 128),
        ("Dense", 1)
    ],
    1: [
        ("GRU", 250),
        ("GRU", 250),
        ("Attention", 128),
        ("Dense", 1)
    ]
}