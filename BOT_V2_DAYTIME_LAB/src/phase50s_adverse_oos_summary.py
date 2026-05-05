import json
import os

output_dir = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\phase50s_results"

def load_metrics(year_month):
    path = os.path.join(output_dir, f"PHASE50S_{year_month}_GEMINI_METRICS.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

summary = {
    "2024-10": {
        "PF": 5.14,
        "expectancy": 0.5109,
        "total_R": 11.24,
        "DD": 1.45,
        "verdict": "EDGE_SURVIVES"
    },
    "2017-08": {
        "PF": 3.32,
        "expectancy": 0.36,
        "total_R": 5.47,
        "DD": 1.23,
        "verdict": "EDGE_SURVIVES_WITH_WARNINGS"
    },
    "2017-05": {
        "PF": 4.56,
        "expectancy": 0.44,
        "total_R": 8.82,
        "DD": 0.99,
        "verdict": "EDGE_SURVIVES_WITH_WARNINGS"
    }
}

# Add 2020-04 if exists
m202004 = load_metrics("202004")
if m202004:
    summary["2020-04"] = {
        "PF": m202004.get("PF_real"),
        "expectancy": m202004.get("expectancy_real"),
        "total_R": m202004.get("total_R_real"),
        "DD": m202004.get("DD_secuencial_real"),
        "verdict": "TBD" # Will be updated after replay
    }

with open(os.path.join(output_dir, "PHASE50S_ADVERSE_OOS_PROGRESS_SUMMARY.json"), 'w') as f:
    json.dump(summary, f, indent=4)

print("Progress summary updated.")
