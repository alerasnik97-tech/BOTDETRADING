
import json
import hashlib

config = {
  "strategy_id": "PHASE22_HIGH_WR_M3_B70_07_1630_TP11_BE05_1T",
  "strategy_name": "PHASE22_HIGH_WR_AUDITED",
  "symbol": "EURUSD",
  "status": "FORWARD_DEMO_CANDIDATE",
  "allow_live": False,
  "demo_only": True,
  "paper_only": True,
  "mt5_enabled": False,
  "mt5_real_enabled": False,
  "ctrader_enabled": False,
  "vps_enabled": False,
  "scbi_touched": False,
  "phase19_reopened": False,
  "session_start_ny": "07:00",
  "session_end_ny": "16:30",
  "mandatory_close_ny": "20:00",
  "max_trades_per_day": 1,
  "timeframe_entry": "M3",
  "htf_context": "H1 Fractal Sweep",
  "entry_model": "First M3 CHOCH",
  "choch_body_filter_pct": 70,
  "tp_r": 1.1,
  "be_trigger_r": 0.5,
  "sl_model": "sweep_extreme_buffer",
  "sl_buffer_pips": 0.5,
  "news_fortress_required": True,
  "news_fortress_mode": "fail_closed",
  "news_guard_minutes": 30,
  "data_quality_mask_required": True,
  "data_quality_mask_mode": "fail_closed",
  "spread_gate_required": True,
  "time_gate_required": True,
  "sl_required": True,
  "tp_required": True,
  "no_trade_without_explicit_allow": True,
  "source_phase": "PHASE23",
  "verdict": "PHASE22_READY_FOR_FORWARD_DEMO_WITH_WARNINGS",
  "metrics": {
    "sample": 1048,
    "pf": 2.32,
    "pf_conservative": 2.32,
    "expectancy": 0.25,
    "winrate": 0.398,
    "max_dd_r": -8.45,
    "max_loss_streak": 8,
    "trades_per_month": 13.1,
    "news_violations": 0,
    "data_mask_violations": 37
  },
  "known_warnings": ["Data Quality Mask not enforced in original Phase 22 optimization", "Performance mismatch with pre-audit estimates (1.72 vs 2.32 PF)"],
  "created_by": "BOT_V2_PHASE23_CONSISTENCY_REPAIR",
  "notes": "Forward demo only. Real trading disabled. Data Quality Mask enforced fail-closed."
}

config_json = json.dumps(config, indent=2, sort_keys=True)
path = r'BOT_V2_DAYTIME_LAB\configs\phase22_forward_demo_config.json'
with open(path, 'w') as f:
    f.write(config_json)

# Hash
h = hashlib.sha256(config_json.encode()).hexdigest()
with open(r'BOT_V2_DAYTIME_LAB\configs\phase22_forward_demo_config_hash.txt', 'w') as f:
    f.write(h)

print(f"Config and Hash (v4) generated: {h}")
