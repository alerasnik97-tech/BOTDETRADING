from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from zoneinfo import ZoneInfo


BASE_ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
LAB_ROOT = BASE_ROOT / "BOT_V2_DAYTIME_LAB"
SRC_ROOT = LAB_ROOT / "src"

PHASE56O_ROOT = (
    LAB_ROOT
    / "reports"
    / "manipulante_tick_historical"
    / "phase56o_corrected_full"
)
PHASE56O_CHECKPOINT = PHASE56O_ROOT / "PHASE56O_CORRECTED_FULL_CHECKPOINT.json"
RAW_TRADES = (
    LAB_ROOT
    / "outputs"
    / "phase38_manipulante_deep_explainer"
    / "csv"
    / "phase38_raw_trades_enriched.csv"
)
PHASE29_BASELINE = (
    LAB_ROOT
    / "outputs"
    / "phase29_wr_loss_streak_compression"
    / "baseline_lock"
    / "phase29_phase25_baseline_metrics.csv"
)
TICK_ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_MARKET_DATA\tick\EURUSD\monthly")
OUT_ROOT = (
    LAB_ROOT
    / "reports"
    / "manipulante_tick_historical"
    / "phase61_zero_trust"
)

NY = ZoneInfo("America/New_York")
UTC = timezone.utc

COMMISSION_PIPS_ROUND_TURN = 0.5
SLIPPAGE_PIPS_ENTRY = 0.2
SLIPPAGE_PIPS_EXIT = 0.2
ROLLOVER_SPREAD_MULTIPLIER = 1.5
ROLLOVER_EXTRA_MULTIPLIER = ROLLOVER_SPREAD_MULTIPLIER - 1.0
ROLL_START = time(16, 0)
ROLL_END = time(20, 30)
PF_CERTIFIED_THRESHOLD = 1.8
PF_REJECT_THRESHOLD = 1.5


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def sha256_file(path: Path, max_bytes: int | None = None) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    total = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            if max_bytes is not None and total + len(chunk) > max_bytes:
                h.update(chunk[: max_bytes - total])
                return f"PARTIAL_{max_bytes}_BYTES_{h.hexdigest()}"
            h.update(chunk)
            total += len(chunk)
    return h.hexdigest()


def round4(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return v
    return round(v, 4)


def pct(value: float, total: float) -> float:
    return round((value / total) * 100.0, 4) if total else 0.0


def in_rollover_window(ts: pd.Timestamp) -> bool:
    if pd.isna(ts):
        return False
    t = ts.astimezone(NY).time()
    return ROLL_START <= t <= ROLL_END


def block_label(ts: pd.Timestamp) -> str:
    t = ts.astimezone(NY).time()
    if time(7, 0) <= t < time(12, 0):
        return "Core [07:00-12:00]"
    hour = t.hour
    minute = 0 if t.minute < 30 else 30
    start = time(hour, minute)
    end_hour = hour
    end_minute = minute + 30
    if end_minute >= 60:
        end_hour += 1
        end_minute -= 60
    end = time(end_hour, end_minute)
    if time(12, 0) <= start < time(20, 30):
        return f"[{start.strftime('%H:%M')}-{end.strftime('%H:%M')}]"
    return "OUTSIDE_AUDIT_WINDOW"


def pf_from_series(values: pd.Series) -> float:
    wins = float(values[values > 0].sum())
    losses = abs(float(values[values < 0].sum()))
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return wins / losses


def metrics_from_series(values: pd.Series) -> dict[str, Any]:
    total = int(len(values))
    wins = values[values > 0]
    losses = values[values < 0]
    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    pf = gross_profit / gross_loss if gross_loss else (float("inf") if gross_profit > 0 else 0.0)
    return {
        "trades": total,
        "net_r": round4(values.sum()),
        "gross_profit_r": round4(gross_profit),
        "gross_loss_r": round4(gross_loss),
        "profit_factor": round4(pf),
        "win_rate_pct": round4(pct(len(wins), total)),
        "expectancy_r": round4(values.mean() if total else 0.0),
    }


def file_inventory() -> pd.DataFrame:
    files = [
        ("phase61_script", SRC_ROOT / "phase61_codex_forensic_auditor.py", True),
        ("phase56o_checkpoint", PHASE56O_CHECKPOINT, True),
        ("phase56o_runner", SRC_ROOT / "phase56o_corrected_full_historical_runner.py", True),
        ("phase29_baseline_trade_file", PHASE29_BASELINE, True),
        ("phase29_signal_source", SRC_ROOT / "phase29_wr_loss_streak_compression.py", True),
        ("h1_fractal_source", SRC_ROOT / "phase18_h1_fractal_sweep.py", True),
        ("m3_choch_source", SRC_ROOT / "phase18_first_3m_choch.py", True),
        ("raw_phase38_trades", RAW_TRADES, True),
        ("mt5_time_gate_support", SRC_ROOT / "phase37_ftmo_trial_support.py", True),
    ]
    rows = []
    for name, path, hash_it in files:
        rows.append(
            {
                "name": name,
                "path": str(path),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else None,
                "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()
                if path.exists()
                else None,
                "sha256": sha256_file(path) if hash_it and path.exists() else None,
            }
        )
    tick_files = sorted(TICK_ROOT.glob("EURUSD_ticks_*.parquet"))
    rows.append(
        {
            "name": "tick_parquet_directory",
            "path": str(TICK_ROOT),
            "exists": TICK_ROOT.exists(),
            "bytes": sum(p.stat().st_size for p in tick_files),
            "modified_utc": None,
            "sha256": f"FILE_COUNT_{len(tick_files)}",
        }
    )
    return pd.DataFrame(rows)


def load_phase56o_trades() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cp = read_json(PHASE56O_CHECKPOINT)
    frames: list[pd.DataFrame] = []
    inventory: list[dict[str, Any]] = []
    for item in cp.get("historical_progress", []):
        month = str(item.get("month"))
        status = item.get("status")
        mkey = month.replace("-", "")
        csv_path = PHASE56O_ROOT / f"month_{mkey}" / f"PHASE56O_MONTH_{mkey}_TRADE_LEVEL.csv"
        rows = 0
        if status == "FORENSIC_COMPLETE" and csv_path.exists():
            df_m = pd.read_csv(csv_path)
            rows = len(df_m)
            df_m["source_month"] = month
            df_m["source_file"] = str(csv_path)
            frames.append(df_m)
        inventory.append(
            {
                "month": month,
                "checkpoint_status": status,
                "sample_total": item.get("sample_total"),
                "checkpoint_auditables": item.get("auditables"),
                "checkpoint_non_auditables": item.get("non_auditables"),
                "trade_file": str(csv_path),
                "trade_file_exists": csv_path.exists(),
                "trade_file_rows": rows,
                "row_match": rows == int(item.get("auditables") or 0)
                if status == "FORENSIC_COMPLETE"
                else None,
            }
        )
    if not frames:
        raise RuntimeError("No PHASE56O FORENSIC_COMPLETE trade files found")
    trades = pd.concat(frames, ignore_index=True)
    month_inv = pd.DataFrame(inventory)
    summary = {
        "checkpoint_path": str(PHASE56O_CHECKPOINT),
        "checkpoint_entries": len(cp.get("historical_progress", [])),
        "complete_months": int((month_inv["checkpoint_status"] == "FORENSIC_COMPLETE").sum()),
        "trade_files_loaded": len(frames),
        "trade_rows_loaded": int(len(trades)),
        "checkpoint_sample_total_sum": int(month_inv["sample_total"].fillna(0).sum()),
        "checkpoint_auditables_sum": int(month_inv["checkpoint_auditables"].fillna(0).sum()),
        "checkpoint_non_auditables_sum": int(month_inv["checkpoint_non_auditables"].fillna(0).sum()),
        "all_complete_rows_match": bool(
            month_inv.loc[month_inv["checkpoint_status"] == "FORENSIC_COMPLETE", "row_match"].fillna(False).all()
        ),
    }
    return trades, month_inv, summary


def attach_raw_trade_fields(trades: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(RAW_TRADES).reset_index().rename(columns={"index": "trade_id"})
    keep = [
        "trade_id",
        "type",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "sl",
        "tp",
        "risk",
        "status",
        "be_triggered",
        "outcome",
        "r_result",
        "year_month",
        "hour_ny",
        "weekday",
    ]
    raw = raw[keep].rename(
        columns={
            "type": "raw_direction",
            "entry_time": "raw_entry_time",
            "exit_time": "raw_exit_time",
            "status": "raw_status",
            "r_result": "raw_r_result",
        }
    )
    out = trades.merge(raw, on="trade_id", how="left")
    out["entry_time_utc"] = pd.to_datetime(out["entry_time"], utc=True, format="mixed")
    out["exit_time_utc"] = pd.to_datetime(out["exit_time"], utc=True, format="mixed")
    out["entry_ny"] = out["entry_time_utc"].dt.tz_convert(NY)
    out["exit_ny"] = out["exit_time_utc"].dt.tz_convert(NY)
    out["raw_entry_time_utc"] = pd.to_datetime(out["raw_entry_time"], utc=True, format="mixed")
    out["raw_exit_time_utc"] = pd.to_datetime(out["raw_exit_time"], utc=True, format="mixed")
    out["risk_pips"] = out["risk"].abs() * 10000.0
    out["entry_rollover_window"] = out["entry_time_utc"].apply(in_rollover_window)
    out["exit_rollover_window"] = out["exit_time_utc"].apply(in_rollover_window)
    out["audit_block"] = out["entry_time_utc"].apply(block_label)
    out["entry_time_delta_seconds_vs_raw"] = (
        out["entry_time_utc"] - out["raw_entry_time_utc"]
    ).dt.total_seconds()
    out["exit_time_delta_seconds_vs_raw"] = (
        out["exit_time_utc"] - out["raw_exit_time_utc"]
    ).dt.total_seconds()
    expected_comm = COMMISSION_PIPS_ROUND_TURN / out["risk_pips"]
    out["phase61_commission_r"] = expected_comm
    out["phase61_slippage_r"] = (
        SLIPPAGE_PIPS_ENTRY + SLIPPAGE_PIPS_EXIT
    ) / out["risk_pips"]
    audit = {
        "raw_rows": int(len(raw)),
        "phase56o_rows": int(len(trades)),
        "raw_join_missing": int(out["risk"].isna().sum()),
        "min_risk_pips": round4(out["risk_pips"].min()),
        "median_risk_pips": round4(out["risk_pips"].median()),
        "max_risk_pips": round4(out["risk_pips"].max()),
        "max_abs_entry_time_delta_seconds_vs_raw": round4(
            out["entry_time_delta_seconds_vs_raw"].abs().max()
        ),
        "max_abs_exit_time_delta_seconds_vs_raw": round4(
            out["exit_time_delta_seconds_vs_raw"].abs().max()
        ),
        "phase56o_commission_residual_max_r": round4(
            (out["comm_r"] - expected_comm).abs().max()
        ),
        "entry_rollover_window_trades": int(out["entry_rollover_window"].sum()),
        "exit_rollover_window_trades": int(out["exit_rollover_window"].sum()),
        "either_rollover_window_trades": int(
            (out["entry_rollover_window"] | out["exit_rollover_window"]).sum()
        ),
    }
    return out, audit


def source_causality_audit(audited_trade_ids: set[int]) -> tuple[pd.DataFrame, dict[str, Any]]:
    baseline = pd.read_csv(PHASE29_BASELINE).reset_index().rename(columns={"index": "trade_id"})
    for col in ["entry_time", "sweep_time", "choch_time"]:
        baseline[col] = pd.to_datetime(baseline[col], utc=True, format="mixed")
    baseline["legal_h1_close_utc"] = baseline["sweep_time"] + pd.Timedelta(hours=1)
    baseline["entry_before_legal_h1_close_minutes"] = (
        baseline["legal_h1_close_utc"] - baseline["entry_time"]
    ).dt.total_seconds() / 60.0
    baseline["h1_label_left_causal_violation"] = (
        baseline["entry_before_legal_h1_close_minutes"] > 0
    )
    audited = baseline[baseline["trade_id"].isin(audited_trade_ids)].copy()

    phase29_text = (SRC_ROOT / "phase29_wr_loss_streak_compression.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    h1_text = (SRC_ROOT / "phase18_h1_fractal_sweep.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    m3_text = (SRC_ROOT / "phase18_first_3m_choch.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    support_text = (SRC_ROOT / "phase37_ftmo_trial_support.py").read_text(
        encoding="utf-8", errors="ignore"
    )

    h1_resample_left = (
        'resample("1h")' in phase29_text or "resample('1h')" in phase29_text
    ) and (
        '"timestamp_ny": "first"' in phase29_text
        or "'timestamp_ny': 'first'" in phase29_text
    )
    h1_fractal_delay_present = (
        "for i in range(2*n, size)" in h1_text and "center = i - n" in h1_text
    )
    m3_fractal_delay_present = (
        "for i in range(2*n, size)" in m3_text and "center = i - n" in m3_text
    )
    m3_same_bar_entry = "entry_price" in m3_text and "df_close[search_idx]" in m3_text
    mt5_time_fail_closed = (
        "NO_TRADE_SERVER_TIME_UNVALIDATED" in support_text
        and "now_utc().astimezone(NY)" in support_text
    )

    violations = audited["h1_label_left_causal_violation"]
    audit = {
        "phase29_baseline_rows": int(len(baseline)),
        "audited_rows_matched_to_phase29": int(len(audited)),
        "h1_resample_uses_left_label_timestamp_ny_first": bool(h1_resample_left),
        "h1_fractal_confirmation_delay_present": bool(h1_fractal_delay_present),
        "m3_fractal_confirmation_delay_present": bool(m3_fractal_delay_present),
        "m3_enters_on_choch_close_same_bar": bool(m3_same_bar_entry),
        "mt5_live_time_gate_fails_closed_if_server_time_unvalidated": bool(
            mt5_time_fail_closed
        ),
        "audited_h1_label_left_causal_violations": int(violations.sum()),
        "audited_h1_label_left_causal_violation_pct": pct(int(violations.sum()), len(audited)),
        "audited_non_violating_or_exact_close": int((~violations).sum()),
        "max_minutes_entry_before_legal_h1_close": round4(
            audited.loc[violations, "entry_before_legal_h1_close_minutes"].max()
            if violations.any()
            else 0
        ),
        "mean_minutes_entry_before_legal_h1_close": round4(
            audited.loc[violations, "entry_before_legal_h1_close_minutes"].mean()
            if violations.any()
            else 0
        ),
        "lookahead_bias_detected": bool(h1_resample_left and violations.any()),
        "causal_verdict": "FAIL_H1_SWEEP_KNOWN_BEFORE_H1_BAR_CLOSE"
        if h1_resample_left and violations.any()
        else "PASS_STATIC_CAUSALITY",
    }
    cols = [
        "trade_id",
        "entry_time",
        "sweep_time",
        "choch_time",
        "legal_h1_close_utc",
        "entry_before_legal_h1_close_minutes",
        "type",
        "r_return",
        "sweep_type",
        "sweep_level",
        "h1_label_left_causal_violation",
    ]
    return audited[cols], audit


def timezone_audit(trades: pd.DataFrame) -> dict[str, Any]:
    entry_t = trades["entry_ny"].dt.time
    in_strategy_window = (entry_t >= time(7, 0)) & (entry_t <= time(16, 30))
    offsets = trades["entry_time_utc"].apply(
        lambda ts: ts.astimezone(NY).utcoffset().total_seconds() / 3600.0
    )
    dst_samples = []
    for year in range(2015, 2027):
        for month in [1, 3, 6, 11]:
            probe = pd.Timestamp(year=year, month=month, day=15, hour=12, tz=UTC)
            ny = probe.astimezone(NY)
            dst_samples.append(
                {
                    "utc_probe": probe.isoformat(),
                    "ny_probe": ny.isoformat(),
                    "offset_hours": ny.utcoffset().total_seconds() / 3600.0,
                    "dst_active": bool(ny.dst()),
                }
            )
    return {
        "entry_rows": int(len(trades)),
        "entry_ny_min": trades["entry_ny"].min().isoformat(),
        "entry_ny_max": trades["entry_ny"].max().isoformat(),
        "ny_offset_hours_observed": sorted(round4(x) for x in offsets.dropna().unique()),
        "strategy_window_0700_1630_violations": int((~in_strategy_window).sum()),
        "raw_vs_phase56o_entry_utc_max_abs_seconds": round4(
            trades["entry_time_delta_seconds_vs_raw"].abs().max()
        ),
        "raw_vs_phase56o_exit_utc_max_abs_seconds": round4(
            trades["exit_time_delta_seconds_vs_raw"].abs().max()
        ),
        "dst_reference_samples_2015_2026": dst_samples,
        "utc_to_ny_conversion_verdict": "PASS_ZONEINFO_DST_OFFSETS_PRESENT"
        if sorted(offsets.dropna().unique().round(0).astype(int).tolist()) == [-5, -4]
        else "FAIL_UNEXPECTED_NY_OFFSETS",
    }


@dataclass
class SpreadLookupResult:
    spreads: pd.DataFrame
    stats: dict[str, Any]


def tick_path_for_ts(ts: pd.Timestamp) -> Path:
    ts_utc = ts.astimezone(UTC)
    return TICK_ROOT / f"EURUSD_ticks_{ts_utc.year}_{ts_utc.month:02d}.parquet"


def build_spread_requests(trades: pd.DataFrame) -> pd.DataFrame:
    rollover = trades["entry_rollover_window"] | trades["exit_rollover_window"]
    preliminary_net = (
        trades["gross_r"] - trades["phase61_commission_r"] - trades["phase61_slippage_r"]
    )
    top_trade_ids = set(trades.loc[preliminary_net.sort_values(ascending=False).head(100).index, "trade_id"])
    top_candidates = trades[trades["trade_id"].isin(top_trade_ids)]
    candidates = pd.concat([trades[rollover], top_candidates], ignore_index=True)
    rows = []
    for _, row in candidates.drop_duplicates("trade_id").iterrows():
        need_entry = bool(row["entry_rollover_window"]) or row["trade_id"] in top_trade_ids
        need_exit = bool(row["exit_rollover_window"]) or row["trade_id"] in top_trade_ids
        if need_entry:
            rows.append(
                {
                    "trade_id": int(row["trade_id"]),
                    "event": "entry",
                    "ts_utc": row["entry_time_utc"],
                    "tick_path": str(tick_path_for_ts(row["entry_time_utc"])),
                }
            )
        if need_exit:
            rows.append(
                {
                    "trade_id": int(row["trade_id"]),
                    "event": "exit",
                    "ts_utc": row["exit_time_utc"],
                    "tick_path": str(tick_path_for_ts(row["exit_time_utc"])),
                }
            )
    req = pd.DataFrame(rows).drop_duplicates(["trade_id", "event"]).reset_index(drop=True)
    return req


def parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq  # type: ignore

        return list(pq.read_schema(path).names)
    except Exception:
        sample = pd.read_parquet(path).head(0)
        return list(sample.columns)


def lookup_spreads(requests: pd.DataFrame) -> SpreadLookupResult:
    if requests.empty:
        return SpreadLookupResult(
            spreads=pd.DataFrame(
                columns=[
                    "trade_id",
                    "event",
                    "requested_ts_utc",
                    "tick_ts_utc",
                    "spread_pips",
                    "lookup_lag_seconds",
                    "bid",
                    "ask",
                    "lookup_status",
                ]
            ),
            stats={"requests": 0, "failures": 0},
        )
    out_frames: list[pd.DataFrame] = []
    failures = []
    grouped = requests.copy()
    grouped["tick_path"] = grouped["tick_path"].astype(str)
    for tick_path_str, req_g in grouped.groupby("tick_path"):
        tick_path = Path(tick_path_str)
        if not tick_path.exists():
            missing = req_g.copy()
            missing["requested_ts_utc"] = missing["ts_utc"]
            missing["tick_ts_utc"] = pd.NaT
            missing["spread_pips"] = math.nan
            missing["lookup_lag_seconds"] = math.nan
            missing["bid"] = math.nan
            missing["ask"] = math.nan
            missing["lookup_status"] = "MISSING_TICK_PARQUET"
            out_frames.append(missing)
            failures.append({"tick_path": tick_path_str, "reason": "MISSING_TICK_PARQUET", "requests": len(req_g)})
            continue
        try:
            cols = parquet_columns(tick_path)
            ts_col = "timestamp_utc" if "timestamp_utc" in cols else "timestamp"
            ticks = pd.read_parquet(tick_path, columns=[ts_col, "bid", "ask"])
            ticks = ticks.rename(columns={ts_col: "tick_ts_utc"})
            ticks["tick_ts_utc"] = pd.to_datetime(ticks["tick_ts_utc"], utc=True, format="mixed")
            ticks = ticks.sort_values("tick_ts_utc")
            ticks["spread_pips"] = (ticks["ask"] - ticks["bid"]) * 10000.0
            req = req_g.rename(columns={"ts_utc": "requested_ts_utc"}).sort_values("requested_ts_utc")
            merged = pd.merge_asof(
                req,
                ticks[["tick_ts_utc", "bid", "ask", "spread_pips"]],
                left_on="requested_ts_utc",
                right_on="tick_ts_utc",
                direction="nearest",
                tolerance=pd.Timedelta(seconds=120),
            )
            merged["lookup_lag_seconds"] = (
                merged["tick_ts_utc"] - merged["requested_ts_utc"]
            ).dt.total_seconds().abs()
            bad = merged["spread_pips"].isna() | (merged["spread_pips"] < 0)
            merged["lookup_status"] = "OK"
            merged.loc[bad, "lookup_status"] = "NO_TICK_WITHIN_120S_OR_BAD_SPREAD"
            out_frames.append(merged)
        except Exception as exc:
            failed = req_g.copy()
            failed["requested_ts_utc"] = failed["ts_utc"]
            failed["tick_ts_utc"] = pd.NaT
            failed["spread_pips"] = math.nan
            failed["lookup_lag_seconds"] = math.nan
            failed["bid"] = math.nan
            failed["ask"] = math.nan
            failed["lookup_status"] = f"PARQUET_READ_ERROR: {exc}"
            out_frames.append(failed)
            failures.append({"tick_path": tick_path_str, "reason": str(exc), "requests": len(req_g)})
    spreads = pd.concat(out_frames, ignore_index=True)
    fail_count = int((spreads["lookup_status"] != "OK").sum())
    stats = {
        "requests": int(len(requests)),
        "unique_tick_files_requested": int(requests["tick_path"].nunique()),
        "ok": int((spreads["lookup_status"] == "OK").sum()),
        "failures": fail_count,
        "failure_details": failures[:20],
        "max_lookup_lag_seconds": round4(spreads["lookup_lag_seconds"].max()),
        "max_spread_pips": round4(spreads["spread_pips"].max()),
        "median_spread_pips": round4(spreads["spread_pips"].median()),
    }
    return SpreadLookupResult(spreads=spreads, stats=stats)


def apply_phase61_friction(trades: pd.DataFrame, spreads: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    piv = spreads.pivot_table(
        index="trade_id",
        columns="event",
        values=["spread_pips", "lookup_lag_seconds"],
        aggfunc="first",
    )
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()
    out = out.merge(piv, on="trade_id", how="left")
    for col in [
        "spread_pips_entry",
        "spread_pips_exit",
        "lookup_lag_seconds_entry",
        "lookup_lag_seconds_exit",
    ]:
        if col not in out.columns:
            out[col] = math.nan
    out["entry_rollover_spread_extra_r"] = 0.0
    out["exit_rollover_spread_extra_r"] = 0.0
    entry_mask = out["entry_rollover_window"]
    exit_mask = out["exit_rollover_window"]
    out.loc[entry_mask, "entry_rollover_spread_extra_r"] = (
        ROLLOVER_EXTRA_MULTIPLIER
        * out.loc[entry_mask, "spread_pips_entry"].fillna(0.0)
        / out.loc[entry_mask, "risk_pips"]
    )
    out.loc[exit_mask, "exit_rollover_spread_extra_r"] = (
        ROLLOVER_EXTRA_MULTIPLIER
        * out.loc[exit_mask, "spread_pips_exit"].fillna(0.0)
        / out.loc[exit_mask, "risk_pips"]
    )
    out["phase61_rollover_spread_extra_r"] = (
        out["entry_rollover_spread_extra_r"] + out["exit_rollover_spread_extra_r"]
    )
    out["phase61_total_friction_r"] = (
        out["phase61_commission_r"]
        + out["phase61_slippage_r"]
        + out["phase61_rollover_spread_extra_r"]
    )
    out["phase61_real_net_r"] = out["gross_r"] - out["phase61_total_friction_r"]
    out["phase61_winner"] = out["phase61_real_net_r"] > 0
    out["phase61_formula"] = "gross_r - commission_r - slippage_r - rollover_spread_extra_r"
    return out


def critical_spread_failures(spreads: pd.DataFrame, trades: pd.DataFrame) -> int:
    if spreads.empty:
        return 0
    flags = trades[["trade_id", "entry_rollover_window", "exit_rollover_window"]].copy()
    merged = spreads.merge(flags, on="trade_id", how="left")
    failed = merged["lookup_status"] != "OK"
    critical_entry = (merged["event"] == "entry") & merged["entry_rollover_window"].fillna(False)
    critical_exit = (merged["event"] == "exit") & merged["exit_rollover_window"].fillna(False)
    return int((failed & (critical_entry | critical_exit)).sum())


def heatmap_30m(trades: pd.DataFrame) -> pd.DataFrame:
    blocks = ["Core [07:00-12:00]"]
    start_h, start_m = 12, 0
    while (start_h, start_m) < (20, 30):
        end_h, end_m = start_h, start_m + 30
        if end_m >= 60:
            end_h += 1
            end_m -= 60
        blocks.append(f"[{start_h:02d}:{start_m:02d}-{end_h:02d}:{end_m:02d}]")
        start_h, start_m = end_h, end_m

    rows = []
    for label in blocks:
        g = trades[trades["audit_block"] == label]
        m = metrics_from_series(g["phase61_real_net_r"]) if not g.empty else {
            "trades": 0,
            "net_r": 0.0,
            "gross_profit_r": 0.0,
            "gross_loss_r": 0.0,
            "profit_factor": 0.0,
            "win_rate_pct": 0.0,
            "expectancy_r": 0.0,
        }
        pf = float(m["profit_factor"])
        net = float(m["net_r"])
        if m["trades"] == 0:
            status = "NO_TRADES"
        elif net <= 0 or pf < 1.0:
            status = "ALPHA_DECAY_NEGATIVE"
        elif pf < PF_REJECT_THRESHOLD:
            status = "FRAGILE_BELOW_1_5"
        elif pf < PF_CERTIFIED_THRESHOLD:
            status = "SURVIVES_BELOW_1_8"
        else:
            status = "EFFICIENT_ABOVE_1_8"
        rows.append(
            {
                "Bloque": label,
                "Trades": m["trades"],
                "Net_R": m["net_r"],
                "PF_Castigado": m["profit_factor"],
                "Win_Rate_Real": m["win_rate_pct"],
                "Expectancy_R": m["expectancy_r"],
                "Status": status,
            }
        )
    return pd.DataFrame(rows)


def source_inventory_outputs(
    file_inv: pd.DataFrame, month_inv: pd.DataFrame, spread_stats: dict[str, Any]
) -> pd.DataFrame:
    rows = file_inv.copy()
    rows["certification_status"] = rows["exists"].map(lambda x: "PRESENT" if x else "MISSING")
    rows.loc[rows["name"] == "tick_parquet_directory", "certification_status"] = (
        f"PRESENT_FOR_SPREAD_LOOKUP_{spread_stats.get('unique_tick_files_requested', 0)}_FILES_REQUESTED"
    )
    rows.loc[len(rows)] = {
        "name": "phase56o_monthly_trade_files",
        "path": str(PHASE56O_ROOT),
        "exists": True,
        "bytes": None,
        "modified_utc": None,
        "sha256": f"MONTHS_{int((month_inv['checkpoint_status'] == 'FORENSIC_COMPLETE').sum())}_ROWS_{int(month_inv['trade_file_rows'].sum())}",
        "certification_status": "MONTHLY_ROW_COUNTS_MATCH"
        if bool(month_inv.loc[month_inv["checkpoint_status"] == "FORENSIC_COMPLETE", "row_match"].fillna(False).all())
        else "MONTHLY_ROW_COUNT_MISMATCH",
    }
    return rows


def top_miracle_trades(trades: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "trade_id",
        "entry_time_utc",
        "entry_ny",
        "exit_time_utc",
        "exit_ny",
        "direction",
        "verdict",
        "gross_r",
        "phase61_real_net_r",
        "risk_pips",
        "spread_pips_entry",
        "spread_pips_exit",
        "phase61_commission_r",
        "phase61_slippage_r",
        "phase61_rollover_spread_extra_r",
        "phase61_total_friction_r",
    ]
    out = trades.sort_values("phase61_real_net_r", ascending=False).head(5).copy()
    out["execution_physical_status"] = out.apply(
        lambda r: "PASS_TICK_SPREAD_OBSERVED"
        if (
            pd.notna(r.get("spread_pips_entry"))
            and pd.notna(r.get("spread_pips_exit"))
            and r.get("spread_pips_entry") >= 0
            and r.get("spread_pips_exit") >= 0
        )
        else "INCONCLUSIVE_SPREAD_LOOKUP_NOT_REQUIRED_OR_MISSING",
        axis=1,
    )
    out["signal_causality_status"] = "REJECTED_IF_H1_LABEL_LEFT_VIOLATION"
    return out[cols + ["execution_physical_status", "signal_causality_status"]]


def make_markdown_report(
    verdict: dict[str, Any],
    friction: dict[str, Any],
    heatmap: pd.DataFrame,
    causal: dict[str, Any],
    miracles: pd.DataFrame,
) -> str:
    def simple_markdown_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "(sin filas)"
        cols = list(df.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in df.iterrows():
            vals = [str(row[col]) for col in cols]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    heatmap_md = simple_markdown_table(heatmap)
    miracle_md = miracles.copy()
    for col in ["entry_time_utc", "entry_ny", "exit_time_utc", "exit_ny"]:
        miracle_md[col] = miracle_md[col].astype(str)
    miracle_text = simple_markdown_table(miracle_md)
    reasons = "\n".join(f"- {r}" for r in verdict["rejection_reasons"])
    return f"""# PHASE61_FINAL_VERDICT

## Veredicto de Veracidad

Binary Status: **{verdict['binary_status']}**

PF_Real: **{verdict['pf_real']}**

Decision: **{verdict['decision_string']}**

Razones:
{reasons}

## Tabla de Friccion

| Concepto | R evaporado | USD por 1 lote estandar |
|---|---:|---:|
| Comision FTMO $5/lot round-turn | {friction['commission_r']} | {friction['commission_usd_per_1lot']} |
| Slippage punitivo 0.2+0.2 pips | {friction['slippage_r']} | {friction['slippage_usd_per_1lot']} |
| Extra spread rollover 1.5x | {friction['rollover_spread_extra_r']} | {friction['rollover_extra_usd_per_1lot']} |
| Total friccion Phase61 | {friction['total_friction_r']} | {friction['total_usd_per_1lot']} |

Gross_R antes de friccion Phase61: {friction['gross_r_before_phase61']}
Net_R Phase56O FTMO previo: {friction['phase56o_net_r']}
Net_R Phase61 castigado: {friction['phase61_real_net_r']}

## Mapa de Calor de Eficiencia 30-min

{heatmap_md}

## Certificacion de Horario

Dictamen: **{verdict['schedule_certification']}**

Operar o extender autoridad hacia 12:00-20:30 NY no queda certificado. La franja tarde muestra bloques fragiles/negativos y, mas importante, el motor historico queda rechazado por sesgo causal H1. Cualquier horario posterior debe revalidarse desde cero con H1 legalmente cerrado.

## Auditoria Time-Sync y Causalidad

- UTC->NY historico: {verdict['timezone_verdict']}
- H1 resample label-left detectado: {causal['h1_resample_uses_left_label_timestamp_ny_first']}
- Trades auditables con entrada antes del cierre legal H1: {causal['audited_h1_label_left_causal_violations']} / {causal['audited_rows_matched_to_phase29']} ({causal['audited_h1_label_left_causal_violation_pct']}%)
- Media de minutos anticipados: {causal['mean_minutes_entry_before_legal_h1_close']}
- Maximo de minutos anticipados: {causal['max_minutes_entry_before_legal_h1_close']}
- Causal verdict: {causal['causal_verdict']}

## Trades Milagrosos

{miracle_text}
"""


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    file_inv = file_inventory()
    trades, month_inv, phase56_summary = load_phase56o_trades()
    trades, raw_audit = attach_raw_trade_fields(trades)
    causal_rows, causal_audit = source_causality_audit(set(trades["trade_id"].astype(int)))
    tz_audit = timezone_audit(trades)

    spread_requests = build_spread_requests(trades)
    spread_lookup = lookup_spreads(spread_requests)
    stressed = apply_phase61_friction(trades, spread_lookup.spreads)
    spread_critical_failures = critical_spread_failures(spread_lookup.spreads, stressed)

    heatmap = heatmap_30m(stressed)
    miracles = top_miracle_trades(stressed)

    gross_metrics = metrics_from_series(stressed["gross_r"])
    phase56_net_metrics = metrics_from_series(stressed["net_r"])
    phase61_metrics = metrics_from_series(stressed["phase61_real_net_r"])

    spread_extra_pips = (
        stressed.loc[stressed["entry_rollover_window"], "spread_pips_entry"].fillna(0.0).sum()
        + stressed.loc[stressed["exit_rollover_window"], "spread_pips_exit"].fillna(0.0).sum()
    ) * ROLLOVER_EXTRA_MULTIPLIER
    friction = {
        "commission_r": round4(stressed["phase61_commission_r"].sum()),
        "slippage_r": round4(stressed["phase61_slippage_r"].sum()),
        "rollover_spread_extra_r": round4(stressed["phase61_rollover_spread_extra_r"].sum()),
        "total_friction_r": round4(stressed["phase61_total_friction_r"].sum()),
        "gross_r_before_phase61": round4(stressed["gross_r"].sum()),
        "phase56o_net_r": round4(stressed["net_r"].sum()),
        "phase61_real_net_r": round4(stressed["phase61_real_net_r"].sum()),
        "commission_usd_per_1lot": round4(len(stressed) * 5.0),
        "slippage_usd_per_1lot": round4(len(stressed) * (SLIPPAGE_PIPS_ENTRY + SLIPPAGE_PIPS_EXIT) * 10.0),
        "rollover_extra_usd_per_1lot": round4(spread_extra_pips * 10.0),
        "total_usd_per_1lot": round4(
            len(stressed) * 5.0
            + len(stressed) * (SLIPPAGE_PIPS_ENTRY + SLIPPAGE_PIPS_EXIT) * 10.0
            + spread_extra_pips * 10.0
        ),
    }

    rejection_reasons = []
    if causal_audit["lookahead_bias_detected"]:
        rejection_reasons.append(
            "LOOKAHEAD_BIAS_H1_LABEL_LEFT: H1 sweep uses the hour-open label while high/low/close are only known at hour close."
        )
    if phase61_metrics["profit_factor"] < PF_REJECT_THRESHOLD:
        rejection_reasons.append("PF_REAL_BELOW_1_5")
    if spread_critical_failures:
        rejection_reasons.append("CRITICAL_ROLLOVER_SPREAD_LOOKUP_FAILURES_FAIL_CLOSED")
    if raw_audit["raw_join_missing"]:
        rejection_reasons.append("RAW_TRADE_JOIN_MISSING_FAIL_CLOSED")
    if not phase56_summary["all_complete_rows_match"]:
        rejection_reasons.append("PHASE56O_MONTHLY_ROW_MISMATCH_FAIL_CLOSED")

    binary_status = (
        "CERTIFIED"
        if (
            phase61_metrics["profit_factor"] > PF_CERTIFIED_THRESHOLD
            and not rejection_reasons
        )
        else "REJECTED"
    )
    afternoon = heatmap[heatmap["Bloque"] != "Core [07:00-12:00]"]
    first_decay = afternoon[
        afternoon["Status"].isin(["ALPHA_DECAY_NEGATIVE", "FRAGILE_BELOW_1_5"])
    ].head(1)
    first_decay_block = (
        str(first_decay.iloc[0]["Bloque"]) if not first_decay.empty else "NONE_DETECTED"
    )
    schedule_certification = (
        "NEGLIGENCIA_FINANCIERA_NO_CERTIFICABLE"
        if binary_status == "REJECTED"
        else "PROFESSIONAL_ONLY_IF_LIMITED_TO_CERTIFIED_BLOCKS"
    )
    verdict = {
        "phase": "PHASE61_FINAL_VERDICT",
        "binary_status": binary_status,
        "decision_string": f"{binary_status}_PF_REAL_{phase61_metrics['profit_factor']}",
        "pf_real": phase61_metrics["profit_factor"],
        "pf_gross_before_phase61": gross_metrics["profit_factor"],
        "pf_phase56o_net_prior": phase56_net_metrics["profit_factor"],
        "total_trades_audited": phase61_metrics["trades"],
        "sample_total_raw_phase38": raw_audit["raw_rows"],
        "phase56o_non_auditables": phase56_summary["checkpoint_non_auditables_sum"],
        "real_win_rate_pct": phase61_metrics["win_rate_pct"],
        "real_expectancy_r": phase61_metrics["expectancy_r"],
        "first_afternoon_decay_block": first_decay_block,
        "schedule_certification": schedule_certification,
        "timezone_verdict": tz_audit["utc_to_ny_conversion_verdict"],
        "lookahead_bias_detected": causal_audit["lookahead_bias_detected"],
        "rejection_reasons": rejection_reasons,
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }

    source_inv = source_inventory_outputs(file_inv, month_inv, spread_lookup.stats)

    out_trade = stressed.copy()
    for col in ["entry_time_utc", "exit_time_utc", "entry_ny", "exit_ny"]:
        out_trade[col] = out_trade[col].astype(str)
    out_trade.to_csv(OUT_ROOT / "PHASE61_TRADE_LEVEL_STRESS.csv", index=False)
    heatmap.to_csv(OUT_ROOT / "PHASE61_HEATMAP_30MIN.csv", index=False)
    miracles.to_csv(OUT_ROOT / "PHASE61_TOP5_MIRACLE_TRADES.csv", index=False)
    source_inv.to_csv(OUT_ROOT / "PHASE61_SOURCE_INVENTORY.csv", index=False)
    month_inv.to_csv(OUT_ROOT / "PHASE61_PHASE56O_MONTH_INVENTORY.csv", index=False)
    causal_rows.to_csv(OUT_ROOT / "PHASE61_CAUSALITY_TRADE_AUDIT.csv", index=False)
    spread_lookup.spreads.to_csv(OUT_ROOT / "PHASE61_SPREAD_LOOKUP_EVENTS.csv", index=False)

    full_report = {
        "verdict": verdict,
        "friction_table": friction,
        "metrics": {
            "gross_before_phase61": gross_metrics,
            "phase56o_net_prior": phase56_net_metrics,
            "phase61_real": phase61_metrics,
        },
        "source_integrity": {
            "phase56o": phase56_summary,
            "raw_trade_join": raw_audit,
            "spread_lookup": {
                **spread_lookup.stats,
                "critical_rollover_spread_failures": spread_critical_failures,
            },
        },
        "time_sync_master": tz_audit,
        "causality_filter": causal_audit,
        "required_parameters": {
            "commission_pips_round_turn": COMMISSION_PIPS_ROUND_TURN,
            "commission_usd_per_standard_lot": 5.0,
            "slippage_entry_pips": SLIPPAGE_PIPS_ENTRY,
            "slippage_exit_pips": SLIPPAGE_PIPS_EXIT,
            "rollover_spread_multiplier_1600_2030_ny": ROLLOVER_SPREAD_MULTIPLIER,
        },
        "artifacts": {
            "json": str(OUT_ROOT / "PHASE61_FINAL_VERDICT.json"),
            "markdown": str(OUT_ROOT / "PHASE61_FINAL_VERDICT.md"),
            "heatmap_csv": str(OUT_ROOT / "PHASE61_HEATMAP_30MIN.csv"),
            "trade_level_csv": str(OUT_ROOT / "PHASE61_TRADE_LEVEL_STRESS.csv"),
            "miracle_trades_csv": str(OUT_ROOT / "PHASE61_TOP5_MIRACLE_TRADES.csv"),
            "source_inventory_csv": str(OUT_ROOT / "PHASE61_SOURCE_INVENTORY.csv"),
            "causality_csv": str(OUT_ROOT / "PHASE61_CAUSALITY_TRADE_AUDIT.csv"),
            "spread_lookup_csv": str(OUT_ROOT / "PHASE61_SPREAD_LOOKUP_EVENTS.csv"),
        },
    }
    write_json(OUT_ROOT / "PHASE61_FINAL_VERDICT.json", full_report)

    md = make_markdown_report(verdict, friction, heatmap, causal_audit, miracles)
    (OUT_ROOT / "PHASE61_FINAL_VERDICT.md").write_text(md, encoding="utf-8")

    print(json.dumps(verdict, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
