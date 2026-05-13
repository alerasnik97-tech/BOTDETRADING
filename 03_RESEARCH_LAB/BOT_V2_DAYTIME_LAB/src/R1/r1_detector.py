from __future__ import annotations
import pandas as pd
import numpy as np

class R1AbsorptionDetector:
    def __init__(self, wick_to_body_min=2.0, close_back_inside=True):
        self.wick_to_body_min = wick_to_body_min
        self.close_back_inside = close_back_inside

    def detect_signals(self, bars: pd.DataFrame, levels: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        if df.index.tz is None: df.index = df.index.tz_localize("UTC")
        df["timestamp_ny"] = df.index.tz_convert("America/New_York")
        df["date_ny"] = df["timestamp_ny"].dt.date
        df = df.join(levels, on="date_ny")
        
        signals = []
        df["body"] = (df["close"] - df["open"]).abs()
        df["wick_top"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["wick_bottom"] = df[["open", "close"]].min(axis=1) - df["low"]
        
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs()
        ], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()

        for row in df.itertuples():
            level_targets = [
                ("pdh", "long_rejection"), ("pdl", "short_rejection"),
                ("asia_h", "long_rejection"), ("asia_l", "short_rejection"),
                ("london_h", "long_rejection"), ("london_l", "short_rejection"),
                ("ny_pre_h", "long_rejection"), ("ny_pre_l", "short_rejection")
            ]
            
            for level_col, mode in level_targets:
                lv = getattr(row, level_col, None)
                if lv is None or np.isnan(lv): continue
                
                if mode == "long_rejection" and row.high > lv:
                    ratio = row.wick_top / row.body if row.body > 0 else row.wick_top / 0.00001
                    if ratio >= self.wick_to_body_min and (not self.close_back_inside or row.close <= lv):
                        signals.append(self._make_sig(row, lv, level_col, "SHORT", ratio))
                        
                elif mode == "short_rejection" and row.low < lv:
                    ratio = row.wick_bottom / row.body if row.body > 0 else row.wick_bottom / 0.00001
                    if ratio >= self.wick_to_body_min and (not self.close_back_inside or row.close >= lv):
                        signals.append(self._make_sig(row, lv, level_col, "LONG", ratio))
                                
        return pd.DataFrame(signals)

    def _make_sig(self, row, level_val, level_type, direction, ratio):
        return {
            "timestamp_utc": row.Index, "timestamp_ny": row.timestamp_ny,
            "level_type": level_type, "level_val": level_val, "direction": direction,
            "wick_to_body": ratio, "wick_top": row.wick_top, "wick_bottom": row.wick_bottom,
            "body": row.body, "atr": row.atr, "high": row.high, "low": row.low,
            "close": row.close, "open": row.open
        }
