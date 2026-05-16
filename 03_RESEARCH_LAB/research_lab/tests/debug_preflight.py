import pandas as pd
import numpy as np

def test():
    idx = pd.date_range('2024-12-30', periods=10, freq='D', tz='UTC')
    cutoff_ts = pd.Timestamp("2025-01-01", tz="UTC")
    mask = idx >= cutoff_ts
    print(f"Index: {idx}")
    print(f"Cutoff: {cutoff_ts}")
    print(f"Mask: {mask}")
    print(f"Any: {mask.any()}")
    if mask.any():
        print("RAISING!")
        raise RuntimeError("BOOM")

if __name__ == "__main__":
    test()
