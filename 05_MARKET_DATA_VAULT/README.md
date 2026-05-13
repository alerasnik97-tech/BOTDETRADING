# 05 MARKET DATA VAULT
**Status:** INSTITUTIONAL STORAGE (Local Only)

## Data Policy
The core market data (Parquet, Tick Data, Raw CSVs) is strictly excluded from version control due to:
1. **Size Constraints:** Datasets exceed GitHub's recommended repository limits.
2. **Proprietary nature:** Tick data is high-value intellectual property.
3. **Volatility:** Historical data remains static once certified.

## Available in Git
- `DATA_MANIFEST.csv`: Inventory of certified datasets.
- `SCHEMA.md`: Structural definition of data frames.
- `DATA_POLICY.md`: Governance rules for data handling.

## How to get Data
For cloud execution (Kaggle/Colab), use the official project dataset or the cloud-upload utility located in `08_CLOUD_FREE_RUN_LAB/`.
