import sys
import pandas as pd
from pathlib import Path
from zoneinfo import ZoneInfo

project_root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB")
sys.path.append(str(project_root))

from src.v7_engine.schedule_guard import ScheduleGuard

def probe_timezone():
    # Test with default guard (8-11 NY)
    guard_default = ScheduleGuard(entry_start_hour=8, entry_end_hour=11)
    # Test with extended guard (7-17 NY)
    guard_extended = ScheduleGuard(entry_start_hour=7, entry_end_hour=17)
    
    test_dates = [
        "2022-05-03 03:15:00",
        "2022-05-04 08:30:00",
        "2022-05-04 09:15:00",
        "2022-05-03 11:45:00",
        "2023-01-02 08:50:00"
    ]
    
    results = []
    print("Timezone Probe:")
    for dt_str in test_dates:
        ts_utc = pd.Timestamp(dt_str).replace(tzinfo=ZoneInfo("UTC"))
        ny_time = ts_utc.astimezone(ZoneInfo("America/New_York"))
        
        acc_def = guard_default.is_entry_permitted(ts_utc)
        acc_ext = guard_extended.is_entry_permitted(ts_utc)
        
        results.append({
            "timestamp_input": dt_str,
            "ny_time": ny_time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "inside_08_11_ny": acc_def,
            "inside_07_17_ny": acc_ext,
            "status": "VALID" if (acc_ext and not acc_def) or acc_def else "REJECTED_BOTH"
        })
        print(f"UTC: {dt_str} -> NY: {results[-1]['ny_time']} | Def: {acc_def} | Ext: {acc_ext}")

    df = pd.DataFrame(results)
    df.to_csv(project_root / "reports/v50b_real_schedule_news_wiring_gate/audits/V50B_TIMEZONE_CONSISTENCY_AUDIT.csv", index=False)
    
    with open(project_root / "reports/v50b_real_schedule_news_wiring_gate/audits/V50B_TIMEZONE_CONSISTENCY_AUDIT.md", "w") as f:
        f.write("# V50B TIMEZONE CONSISTENCY AUDIT\n\n")
        f.write("| Timestamp UTC | NY Time | Allowed 08-11 | Allowed 07-17 | Status |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for r in results:
            f.write(f"| {r['timestamp_input']} | {r['ny_time']} | {r['inside_08_11_ny']} | {r['inside_07_17_ny']} | {r['status']} |\n")

if __name__ == "__main__":
    probe_timezone()
