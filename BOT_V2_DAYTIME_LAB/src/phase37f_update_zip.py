import zipfile
import os
from pathlib import Path

ROOT = Path(r"c:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
ZIP_PATH = ROOT / "000_PARA_CHATGPT.zip"

FILES_TO_ADD = [
    "MANIPULANTE/09_COMPLIANCE/MT5_LIVE_NEWS_ADAPTER/MANIPULANTE_CalendarServiceExporter.mq5",
    "MANIPULANTE/09_COMPLIANCE/MT5_LIVE_NEWS_ADAPTER/MANIPULANTE_CalendarBootstrapEA.mq5",
    "BOT_V2_DAYTIME_LAB/reports/PHASE37F_MQL5_CALENDAR_SERVICE_EXPORTER_REPORT.md"
]

def update_zip():
    with zipfile.ZipFile(ZIP_PATH, 'a') as zf:
        for f in FILES_TO_ADD:
            full_path = ROOT / f
            if full_path.exists():
                zf.write(full_path, f)
                print(f"Added {f}")
            else:
                print(f"Warning: {f} not found")

if __name__ == "__main__":
    update_zip()
