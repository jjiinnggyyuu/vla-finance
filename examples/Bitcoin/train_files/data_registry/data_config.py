"""Bitcoin 1h OHLCVA experiment notes.

The Bitcoin prototype uses `starVLA.dataloader.bitcoin_datasets`, not the
LeRobot registry. Keep the experiment constants here so the YAML and any
future evaluation scripts share the same assumptions.
"""

INPUT_WINDOW = 80
PREDICTION_HORIZON = 12
ACTION_DIM = 4  # open, high, low, close
TARGET_MODE = "log_ratio"  # target = log(future_ohlc / last_input_close)

SPLITS = {
    "train": ("2020-01-01", "2024-06-30"),
    "validation": ("2024-07-01", "2025-03-31"),
    "test": ("2025-04-01", "2026-03-31"),
}

COLUMNS = {
    "timestamp": "timestamp",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "amount": "quote_volume",
}
