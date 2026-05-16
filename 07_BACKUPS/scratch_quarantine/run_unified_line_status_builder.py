from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scratch.unified_line_status_lib import build_unified_surface, write_unified_surface


def main() -> None:
    surface, csv_frame = build_unified_surface()
    write_unified_surface(surface, csv_frame)
    print(json.dumps(
        {
            "status": "PASS",
            "output_json": "results/SCBI_UNIFIED_LINE_STATUS.json",
            "output_csv": "results/SCBI_UNIFIED_LINE_STATUS.csv",
            "line_count": len(surface["lines"]),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
