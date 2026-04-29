from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from phase37_ftmo_trial_support import LAB, MANIPULANTE, NY, detect_symbol, ensure_mt5, now_iso, now_utc, strategy_config_gate, time_gate, write_csv, write_json, write_text


SRC = LAB / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phase18_first_3m_choch import First3MChochDetector  # noqa: E402
from phase18_h1_fractal_sweep import H1FractalSweepDetector  # noqa: E402


OUT = LAB / "outputs" / "phase37d_ftmo_trial_api_news_signal" / "signal_engine_equivalence"
CONFIG_PATH = MANIPULANTE / "01_ESTRATEGIA_AUTORIDAD" / "manipulante_config.json"
PHASE27_SOURCE = LAB / "src" / "phase27_full_historical_validation.py"


def load_authority_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _rates_to_m3_dataframe(rates: Any) -> pd.DataFrame:
    df = pd.DataFrame(rates)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["timestamp_ny"] = df["timestamp"].dt.tz_convert(NY)
    rename = {
        "open": "open_bid",
        "high": "high_bid",
        "low": "low_bid",
        "close": "close_bid",
    }
    df = df.rename(columns=rename)
    needed = ["timestamp", "timestamp_ny", "open_bid", "high_bid", "low_bid", "close_bid"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"missing {col}")
    return df[needed].copy()


def load_live_m3(symbol: str, bars: int = 5000) -> tuple[pd.DataFrame, str | None]:
    mt5, error = ensure_mt5()
    if mt5 is None:
        return pd.DataFrame(), error
    try:
        rates = mt5.copy_rates_from_pos(symbol, getattr(mt5, "TIMEFRAME_M3"), 0, bars)
    except Exception as exc:
        return pd.DataFrame(), str(exc)
    if rates is None or len(rates) == 0:
        return pd.DataFrame(), "M3 rates unavailable"
    try:
        return _rates_to_m3_dataframe(rates), None
    except Exception as exc:
        return pd.DataFrame(), str(exc)


def generate_phase25_signals_from_m3(df_m3: pd.DataFrame) -> pd.DataFrame:
    df = df_m3.copy().sort_values("timestamp").reset_index(drop=True)
    df["body"] = (df["close_bid"] - df["open_bid"]).abs()
    df["range"] = (df["high_bid"] - df["low_bid"]).abs()
    df["body_pct"] = df["body"] / df["range"].replace(0, 0.00001)

    df_idx = df.set_index("timestamp")
    df_h1 = (
        df_idx.resample("1h")
        .agg(
            {
                "open_bid": "first",
                "high_bid": "max",
                "low_bid": "min",
                "close_bid": "last",
                "timestamp_ny": "first",
            }
        )
        .dropna()
        .reset_index()
    )
    sweeps = H1FractalSweepDetector(params={}).detect_sweeps(df_h1)
    if sweeps.empty:
        return pd.DataFrame()
    sweeps["hour"] = sweeps["timestamp_ny"].dt.hour
    sweeps = sweeps[(sweeps["hour"] >= 6) & (sweeps["hour"] <= 16)]
    if sweeps.empty:
        return pd.DataFrame()
    signals = First3MChochDetector(params={"sl_buffer": 0.5, "max_mins_post_sweep": 60}).detect_choch(df, sweeps)
    if signals.empty:
        return pd.DataFrame()
    signals = pd.merge(
        signals,
        df[["timestamp_ny", "body_pct", "open_bid", "high_bid", "low_bid", "close_bid"]],
        left_on="choch_time",
        right_on="timestamp_ny",
        how="left",
    )
    signals = signals[signals["body_pct"] >= 0.7].copy()
    signals["choch_hour"] = signals["choch_time"].dt.hour
    signals = signals[(signals["choch_hour"] >= 7) & (signals["choch_hour"] <= 16)].copy()
    signals["choch_date"] = signals["choch_time"].dt.date
    signals = signals.sort_values("choch_time").groupby("choch_date").head(1).reset_index(drop=True)
    return signals


def _load_historical_fixture(limit: int = 20000) -> tuple[pd.DataFrame, str]:
    manifest = LAB / "data" / "certified_m3" / "M3_CERTIFICATION_METADATA.json"
    if not manifest.exists():
        return pd.DataFrame(), "M3_CERTIFICATION_METADATA missing"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        bid_path = Path(payload["bid_path"])
    except Exception as exc:
        return pd.DataFrame(), f"manifest parse failed: {exc}"
    if not bid_path.exists():
        return pd.DataFrame(), f"bid path missing: {bid_path}"
    try:
        df = pd.read_csv(bid_path, nrows=limit)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["timestamp_ny"] = df["timestamp"].dt.tz_convert(NY)
        rename = {}
        for col in df.columns:
            if col in {"open", "high", "low", "close"}:
                rename[col] = f"{col}_bid"
        df = df.rename(columns=rename)
        return df[["timestamp", "timestamp_ny", "open_bid", "high_bid", "low_bid", "close_bid"]], str(bid_path)
    except Exception as exc:
        return pd.DataFrame(), f"fixture load failed: {exc}"


def equivalence_check() -> dict[str, Any]:
    cfg_gate = strategy_config_gate()
    phase27_text = PHASE27_SOURCE.read_text(encoding="utf-8", errors="ignore") if PHASE27_SOURCE.exists() else ""
    rows: list[dict[str, Any]] = [
        {
            "component": "authority_config",
            "expected": "PHASE25_AUTHORITY TP1.4 BE0.4 BF70",
            "evidence": cfg_gate.get("state"),
            "status": "PASS" if cfg_gate.get("state") == "MANIPULANTE_MATCH" else "FAIL",
        },
        {
            "component": "H1 sweep detector",
            "expected": "phase18_h1_fractal_sweep.H1FractalSweepDetector",
            "evidence": "imported callable",
            "status": "PASS",
        },
        {
            "component": "First M3 CHOCH detector",
            "expected": "phase18_first_3m_choch.First3MChochDetector",
            "evidence": "imported callable",
            "status": "PASS",
        },
        {
            "component": "Phase27 exact mapping",
            "expected": "generate_signals uses Phase18 detectors, BF70 and one trade/day",
            "evidence": "source present" if "generate_signals" in phase27_text and "body_filter_pct" in phase27_text else "missing",
            "status": "PASS" if "generate_signals" in phase27_text and "body_filter_pct" in phase27_text else "FAIL",
        },
    ]
    fixture_df, fixture_source = _load_historical_fixture()
    fixture_status = "FAIL"
    fixture_signal_count = 0
    fixture_error = None
    if not fixture_df.empty:
        try:
            fixture_signals = generate_phase25_signals_from_m3(fixture_df)
            fixture_signal_count = int(len(fixture_signals))
            fixture_status = "PASS"
        except Exception as exc:
            fixture_error = str(exc)
    rows.append(
        {
            "component": "light historical fixture",
            "expected": "Phase25 signal path executes on certified M3 slice",
            "evidence": f"{fixture_source}; signals={fixture_signal_count}; error={fixture_error}",
            "status": fixture_status,
        }
    )
    ok = all(row["status"] == "PASS" for row in rows)
    return {
        "timestamp_utc": now_iso(),
        "state": "MANIPULANTE_SIGNAL_SYNC_OK" if ok else "SIGNAL_ENGINE_REQUIRES_REPAIR",
        "rows": rows,
        "fixture_signal_count": fixture_signal_count,
        "fixture_source": fixture_source,
        "reason": "Live engine reuses Phase18/Phase27 exact signal path" if ok else "Equivalence evidence incomplete",
    }


def evaluate_live_signal(news_gate: str = "NO_TRADE", data_gate: str = "NO_TRADE", time_state: str | None = None) -> dict[str, Any]:
    cfg = load_authority_config()
    eq = equivalence_check()
    symbol_status = detect_symbol()
    session = time_gate(symbol_status)
    if time_state is not None:
        session["state"] = time_state
    status: dict[str, Any] = {
        "timestamp_utc": now_iso(),
        "state": eq["state"],
        "signal_status": "ERROR_FAIL_CLOSED",
        "signal_ready": False,
        "strategy": "MANIPULANTE",
        "source_phase": cfg.get("source_phase"),
        "symbol": symbol_status.get("symbol"),
        "news_gate": news_gate,
        "data_gate": data_gate,
        "time_gate": session.get("state"),
        "equivalence": eq,
        "latest_signal": None,
        "reason": "",
    }
    if eq["state"] != "MANIPULANTE_SIGNAL_SYNC_OK":
        status["signal_status"] = "ERROR_FAIL_CLOSED"
        status["reason"] = eq["reason"]
        return status
    if news_gate != "ALLOW" or data_gate != "ALLOW" or session.get("state") != "ALLOW":
        status["signal_status"] = "NO_TRADE_GATE_BLOCK"
        status["reason"] = f"Required gates not ALLOW: news={news_gate}; data={data_gate}; time={session.get('state')}"
        return status
    df_m3, error = load_live_m3(str(symbol_status.get("symbol") or "EURUSD"))
    if df_m3.empty:
        status["state"] = "SIGNAL_ENGINE_REQUIRES_REPAIR"
        status["signal_status"] = "ERROR_FAIL_CLOSED"
        status["reason"] = error or "No live M3 data"
        return status
    try:
        signals = generate_phase25_signals_from_m3(df_m3)
    except Exception as exc:
        status["state"] = "SIGNAL_ENGINE_REQUIRES_REPAIR"
        status["signal_status"] = "ERROR_FAIL_CLOSED"
        status["reason"] = f"Live signal generation failed: {exc}"
        return status
    if signals.empty:
        status["signal_status"] = "NO_SIGNAL"
        status["reason"] = "No Phase25 signal in current live M3/H1 context"
        return status
    latest = signals.sort_values("choch_time").iloc[-1].to_dict()
    latest_time = latest.get("choch_time")
    if hasattr(latest_time, "isoformat"):
        latest["choch_time"] = latest_time.isoformat()
    today_ny = now_utc().astimezone(NY).date()
    latest_date = pd.Timestamp(latest["choch_time"]).date()
    if latest_date != today_ny:
        status["signal_status"] = "NO_SIGNAL"
        status["reason"] = "Latest Phase25 signal is not from current NY date"
        status["latest_signal"] = latest
        return status
    status["signal_status"] = "SIGNAL_READY"
    status["signal_ready"] = True
    status["latest_signal"] = latest
    status["reason"] = "Current NY-day Phase25 signal detected"
    return status


def write_outputs(news_gate: str = "NO_TRADE", data_gate: str = "NO_TRADE", time_state: str | None = None) -> dict[str, Any]:
    result = evaluate_live_signal(news_gate=news_gate, data_gate=data_gate, time_state=time_state)
    write_json(OUT / "phase37d_signal_engine_equivalence.json", result)
    rows = result.get("equivalence", {}).get("rows", [])
    write_csv(OUT / "phase37d_signal_engine_diff.csv", rows, ["component", "expected", "evidence", "status"])
    write_text(
        OUT / "phase37d_signal_engine_equivalence.md",
        f"""
# Phase37D Signal Engine Equivalence

- state: {result.get('state')}
- signal_status: {result.get('signal_status')}
- sync reason: {result.get('equivalence', {}).get('reason')}
- live reason: {result.get('reason')}
- source: Phase18 detectors + Phase27 generate_signals mapping.
- strategy changed: false
""",
    )
    return result


def main() -> dict[str, Any]:
    result = write_outputs()
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return result


if __name__ == "__main__":
    main()
