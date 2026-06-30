"""Build aligned 1h market-hours dataset: BTC + SPY + TLT.

- SPY/TLT: yfinance 1h (native :30 ET bars) -> convert to UTC.
- BTC: resample btc_15m.csv into 1h bars anchored at :30 (to match SPY's grid).
- Keep only timestamps present in ALL three assets (US equity regular hours).

Outputs (identical timestamp set across all three):
    crypto_data/btc_market_1h.csv
    crypto_data/spy_1h.csv
    crypto_data/tlt_1h.csv
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf

ROOT = "/home/jingyu/vla-finance"
BTC_15M = f"{ROOT}/crypto_data/btc_15m.csv"
OUT_DIR = f"{ROOT}/crypto_data"

OHLC_AGG = {"open": "first", "high": "max", "low": "min", "close": "last",
            "volume": "sum"}


def load_btc_30min_anchored() -> pd.DataFrame:
    """BTC 15m (UTC) -> 1h bars anchored at :30 (e.g. 13:30 covers 13:30-14:30)."""
    df = pd.read_csv(BTC_15M, parse_dates=["timestamp"])
    df = df.rename(columns={"quote_volume": "amount"})
    df = df.set_index("timestamp").sort_index()
    df.index = df.index.tz_localize("UTC")           # btc_15m is UTC (Binance)
    # offset="30min" => bins start at :30 (13:30, 14:30, ...). label=left keeps :30 stamp.
    out = df.resample("1h", offset="30min", label="left", closed="left").agg(OHLC_AGG)
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out


def fetch_yf_1h(symbol: str) -> pd.DataFrame:
    """yfinance 1h (native :30 ET) -> UTC, columns lowercased to OHLCV."""
    df = yf.download(symbol, period="730d", interval="1h", progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                            "Close": "close", "Volume": "volume"})
    df = df[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index, utc=True)    # ET -> UTC
    df = df.sort_index()
    return df


def main():
    print("[1/4] BTC 15m -> 1h @:30 (UTC) ...")
    btc = load_btc_30min_anchored()
    print(f"      BTC bars: {len(btc)}  [{btc.index.min()} .. {btc.index.max()}]")

    print("[2/4] SPY, TLT 1h (yfinance) -> UTC ...")
    spy = fetch_yf_1h("SPY")
    tlt = fetch_yf_1h("TLT")
    print(f"      SPY bars: {len(spy)}  [{spy.index.min()} .. {spy.index.max()}]")
    print(f"      TLT bars: {len(tlt)}  [{tlt.index.min()} .. {tlt.index.max()}]")

    print("[3/4] Intersect timestamps (all three present) ...")
    common = btc.index.intersection(spy.index).intersection(tlt.index)
    common = common.sort_values()
    print(f"      common bars: {len(common)}  [{common.min()} .. {common.max()}]")

    print("[4/4] Save aligned CSVs ...")
    for name, df in (("btc_market", btc), ("spy", spy), ("tlt", tlt)):
        sub = df.loc[common].copy()
        sub.insert(0, "timestamp", sub.index.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S"))
        if "amount" not in sub.columns:
            sub["amount"] = sub["volume"]
        cols = ["timestamp", "open", "high", "low", "close", "volume", "amount"]
        path = f"{OUT_DIR}/{name}_1h.csv"
        sub[cols].to_csv(path, index=False)
        print(f"      saved {path}  ({len(sub)} rows)")

    # quick sanity: bars per trading day
    days = pd.Series(common).dt.date.nunique()
    print(f"\nDone. {len(common)} aligned bars over {days} trading days "
          f"(~{len(common)/max(days,1):.1f} bars/day).")


if __name__ == "__main__":
    main()
