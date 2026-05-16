from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_LAB = REPO_ROOT / "03_RESEARCH_LAB"
if str(RESEARCH_LAB) not in sys.path:
    sys.path.insert(0, str(RESEARCH_LAB))

from research_lab.data_preparation.eurusd_prepared_ohlcv_builder import main


if __name__ == "__main__":
    raise SystemExit(main())
