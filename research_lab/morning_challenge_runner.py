import pandas as pd
import numpy as np
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from research_lab.config import EngineConfig, NewsConfig, PAIR_META, with_execution_mode
from research_lab.build_am_grade_news_dataset import build_am_grade_news_dataset
from research_lab.data_loader import load_backtest_data_bundle
from research_lab.engine import run_backtest
from research_lab.news_filter import build_entry_block, require_operational_news
from research_lab.strategies.zscore_mean_reversion_pm import NAME

class StrategyMock:
    def __init__(self, name):
        self.NAME = name
        self.WARMUP_BARS = 20

def run_challenge():
    pair = "EURUSD"
    engine_cfg = EngineConfig(risk_pct=0.5, execution_mode="normal_mode")
    am_summary = build_am_grade_news_dataset()
    if am_summary["module_verdict"] != "READY_FOR_STRICT_AM_RESEARCH":
        missing = ", ".join(am_summary.get("critical_missing_families", [])) or "coverage_critical_pending"
        raise RuntimeError(
            "AM challenge blocked by News Fortress. "
            f"Current verdict={am_summary['module_verdict']}. Missing={missing}."
        )
    news_cfg = NewsConfig(
        enabled=True,
        file_path=Path("data/news_eurusd_am_fortress_v3.csv"),
        raw_file_path=Path("data/official_anchors/out/canonical_anchor_events.csv"),
        source_approved=True,
        pre_minutes=30,
        post_minutes=60,
        forced_exit_pre_news=True,
        cancel_pending_pre_news=True,
        pre_news_exit_minutes=10,
    )

    data_bundle = load_backtest_data_bundle(pair, [Path("data_free_2020/prepared"), Path("data_candidates_2022_2025/prepared")], "2020-01-01", "2025-12-31", "normal_mode")
    news_result = require_operational_news(pair, news_cfg, context="morning_challenge_runner")
    news_block = build_entry_block(data_bundle.frame.index, news_result.events, news_cfg)

    sessions = [
        {"name": "PM_Baseline", "start": "11:00", "end": "16:30"},
        {"name": "AM_Challenge", "start": "08:00", "end": "11:00"}
    ]

    results = []
    params_base = {"z_threshold": 2.5, "lookback": 20, "target_rr": 1.5, "min_bar_atr_ratio": 0.8}

    for sess in sessions:
        strat_mock = StrategyMock(NAME + "_" + sess["name"])
        
        def custom_signal(frame, i, p):
            idx = frame.index[i]
            t = idx.hour * 60 + idx.minute
            s_min = int(sess["start"].split(":")[0])*60 + int(sess["start"].split(":")[1])
            e_min = int(sess["end"].split(":")[0])*60 + int(sess["end"].split(":")[1])
            if t < s_min or t >= e_min: return None
            
            lookback = p["lookback"]
            if i < lookback: return None
            slice_close = frame["close"].iloc[i-lookback:i+1]
            std = slice_close.std(ddof=0)
            if std == 0: return None
            z_score = (frame["close"].iat[i] - slice_close.mean()) / std
            
            bar_range = float(frame["high"].iat[i] - frame["low"].iat[i])
            atr_val = float(frame["atr14"].iat[i])
            if atr_val == 0 or (bar_range / atr_val) < p.get("min_bar_atr_ratio", 0.0): return None

            if z_score > p["z_threshold"]:
                return {"direction": "short", "stop_mode": "atr", "stop_atr": 2.0, "target_mode": "rr", "target_rr": p["target_rr"]}
            if z_score < -p["z_threshold"]:
                return {"direction": "long", "stop_mode": "atr", "stop_atr": 2.0, "target_mode": "rr", "target_rr": p["target_rr"]}
            return None

        strat_mock.signal = custom_signal
        res = run_backtest(
            strat_mock,
            data_bundle.frame,
            params_base,
            engine_cfg,
            news_block,
            news_result.enabled,
            news_events=news_result.events,
            news_settings=news_cfg,
        )
        
        trades = res.trades
        pf = (trades[trades['pnl_r'] > 0]['pnl_r'].sum() / abs(trades[trades['pnl_r'] <= 0]['pnl_r'].sum())) if not trades.empty and trades[trades['pnl_r'] <= 0]['pnl_r'].sum() != 0 else 0.0
        
        results.append({"Session": sess["name"], "Trades": len(trades), "PF": round(pf, 2), "Pnl_R": round(trades['pnl_r'].sum(), 2) if not trades.empty else 0})

    print("\n--- RESULTS: MORNING CHALLENGE (08:00 NY) ---")
    print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    run_challenge()
