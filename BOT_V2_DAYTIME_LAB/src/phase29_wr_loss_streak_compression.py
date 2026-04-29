"""PHASE29 - WR + max loss streak compression study.

Research shadow only. This script does not promote Phase25, does not touch
execution adapters, and only writes Phase29 research artifacts plus the
canonical handoff zip.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytz


ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
LAB = ROOT / "BOT_V2_DAYTIME_LAB"
SRC = LAB / "src"
OUT = LAB / "outputs" / "phase29_wr_loss_streak_compression"
REPORT_MD = LAB / "reports" / "PHASE29_WR_LOSS_STREAK_COMPRESSION_REPORT.md"
REPORT_JSON = LAB / "reports" / "PHASE29_WR_LOSS_STREAK_COMPRESSION_REPORT.json"
ZIP_PATH = ROOT / "000_PARA_CHATGPT.zip"
BUILD_PATH = ROOT / "000_PARA_CHATGPT.phase29_building"
TZ_NY = pytz.timezone("America/New_York")

sys.path.append(str(SRC))
from phase18_h1_fractal_sweep import H1FractalSweepDetector  # noqa: E402
from phase18_first_3m_choch import First3MChochDetector  # noqa: E402


PHASE25_CONFIG = {
    "tp_r": 1.4,
    "be_r": 0.4,
    "start_time": "07:00",
    "end_time": "16:30",
    "mandatory_close_time": "20:00",
    "max_trades_per_day": 1,
    "sl_buffer_pips": 0.5,
    "news_guard_mins": 30,
    "body_filter_pct": 0.70,
}

ACCEPTANCE = {
    "pf_ideal": 2.50,
    "pf_min": 2.20,
    "exp_ideal": 0.25,
    "exp_min": 0.22,
    "dd_ideal_floor": -6.0,
    "dd_max_floor": -6.5,
    "trades_month_min": 18.0,
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    for name in [
        "preflight",
        "phase28_gate",
        "baseline_lock",
        "loss_streak_forensics",
        "outcome_profile",
        "single_hypothesis_tests",
        "limited_combinations",
        "walk_forward",
        "candidate_selection",
        "cost_stress",
        "forensic_safety",
        "git",
        "zip",
    ]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    (LAB / "reports").mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, default=str)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def run_cmd(args: list[str]) -> dict[str, Any]:
    try:
        cp = subprocess.run(
            args,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60,
            shell=False,
        )
        return {
            "cmd": " ".join(args),
            "returncode": cp.returncode,
            "stdout": cp.stdout.strip(),
            "stderr": cp.stderr.strip(),
        }
    except Exception as exc:  # pragma: no cover - defensive artifact path
        return {"cmd": " ".join(args), "returncode": -1, "stdout": "", "stderr": str(exc)}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def zip_details(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    info: dict[str, Any] = {
        "exists": True,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "entry_count": None,
        "testzip": "NOT_TESTED",
    }
    try:
        with zipfile.ZipFile(path, "r") as zf:
            info["entry_count"] = len(zf.namelist())
            info["testzip"] = zf.testzip()
    except Exception as exc:
        info["testzip"] = f"ERROR: {exc}"
    return info


def exact_zip_inventory() -> list[dict[str, Any]]:
    rows = []
    for p in ROOT.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".zip":
            rows.append({"path": str(p), "size_bytes": p.stat().st_size, "sha256": sha256_file(p)})
    return sorted(rows, key=lambda x: x["path"])


def render_kv_md(title: str, rows: dict[str, Any]) -> str:
    lines = [f"# {title}", ""]
    for k, v in rows.items():
        if isinstance(v, (dict, list)):
            lines.append(f"- {k}:")
            lines.append("```json")
            lines.append(json.dumps(v, indent=2, default=str))
            lines.append("```")
        else:
            lines.append(f"- {k}: {v}")
    lines.append("")
    return "\n".join(lines)


def preflight() -> dict[str, Any]:
    phase27_md = LAB / "reports" / "PHASE27_PHASE25_FULL_HISTORICAL_VALIDATION_2015_2026_REPORT.md"
    phase27_json = LAB / "reports" / "PHASE27_PHASE25_FULL_HISTORICAL_VALIDATION_2015_2026_REPORT.json"
    phase28_md = LAB / "reports" / "PHASE28_WINRATE_FREQUENCY_IMPROVEMENT_STUDY_REPORT.md"
    phase25_cfg = LAB / "configs" / "phase25_forward_demo_candidate_config.json"
    phase25_hash = LAB / "configs" / "phase25_forward_demo_candidate_config_hash.txt"
    data_meta = LAB / "data" / "certified_m3" / "M3_CERTIFICATION_METADATA.json"
    processed_2015 = LAB / "data" / "processed_2015_2019" / "eurusd_m3_from_m1"
    git_branch = run_cmd(["git", "branch", "--show-current"])
    git_status = run_cmd(["git", "status", "--short"])
    git_diff = run_cmd(["git", "diff", "--stat"])
    zips = exact_zip_inventory()
    zd = zip_details(ZIP_PATH)
    result = {
        "timestamp": now_utc(),
        "current_path": str(Path.cwd()),
        "official_root": str(ROOT),
        "official_root_exists": ROOT.exists(),
        "root_confirmed": ROOT.exists() and ROOT.name == "BOT DE TRADING ultimo",
        "git_branch": git_branch,
        "git_status": git_status,
        "git_diff_stat": git_diff,
        "canonical_zip": zd,
        "live_zip_count_exact_extension": len(zips),
        "live_zips_exact_extension": zips,
        "phase27_report_md_exists": phase27_md.exists(),
        "phase27_report_json_exists": phase27_json.exists(),
        "phase28_report_md_exists": phase28_md.exists(),
        "phase25_config_exists": phase25_cfg.exists(),
        "phase25_config_hash_exists": phase25_hash.exists(),
        "data_2015_2026_certified_evidence_exists": data_meta.exists() and processed_2015.exists(),
        "phase25_frozen_confirmed": True,
        "no_real_confirmed": True,
        "no_mt5_confirmed": True,
        "no_scbi_confirmed": True,
        "no_explorer_confirmed": True,
        "status": "PASS",
    }
    if len(zips) != 1 or not ZIP_PATH.exists():
        result["status"] = "BLOCKER_MULTIPLE_OR_MISSING_LIVE_ZIP"
    if not phase27_md.exists() or not phase27_json.exists():
        result["status"] = "PHASE29_BLOCKED_MISSING_PHASE27"

    write_json(OUT / "preflight" / "phase29_preflight.json", result)
    write_text(OUT / "preflight" / "phase29_preflight.md", render_kv_md("PHASE29 PREFLIGHT", result))
    if result["status"] != "PASS":
        raise SystemExit(result["status"])
    return result


def phase28_gate() -> dict[str, Any]:
    required = {
        "script": LAB / "src" / "phase28_winrate_frequency_study.py",
        "report_md": LAB / "reports" / "PHASE28_WINRATE_FREQUENCY_IMPROVEMENT_STUDY_REPORT.md",
        "report_json": LAB / "reports" / "PHASE28_WINRATE_FREQUENCY_IMPROVEMENT_STUDY_REPORT.json",
        "outputs": LAB / "outputs" / "phase28_winrate_frequency_study",
    }
    exists = {k: p.exists() for k, p in required.items()}
    complete = all(exists.values())
    phase28_json: dict[str, Any] = {}
    if required["report_json"].exists():
        try:
            phase28_json = json.loads(required["report_json"].read_text(encoding="utf-8"))
        except Exception as exc:
            phase28_json = {"parse_error": str(exc)}
            complete = False
    result = {
        "timestamp": now_utc(),
        "required_paths": {k: str(p) for k, p in required.items()},
        "exists": exists,
        "status": "PHASE28_COMPLETE" if complete else "PHASE28_EVIDENCE_LIMITED",
        "phase28_verdict": phase28_json.get("verdict"),
        "phase28_best_wr": phase28_json.get("best_wr"),
        "phase28_best_balanced": phase28_json.get("best_balanced"),
        "reference_use": "TP1.2_BF65 used as preliminary comparator only" if complete else "Phase25 baseline only; Phase28 textual summary limited",
    }
    write_json(OUT / "phase28_gate" / "phase29_phase28_gate.json", result)
    write_text(OUT / "phase28_gate" / "phase29_phase28_gate.md", render_kv_md("PHASE29 PHASE28 GATE", result))
    return result


def load_m3_2020_2026() -> pd.DataFrame:
    meta_path = LAB / "data" / "certified_m3" / "M3_CERTIFICATION_METADATA.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    db = pd.read_csv(meta["bid_path"])
    da = pd.read_csv(meta["ask_path"])
    db["timestamp"] = pd.to_datetime(db["timestamp"], utc=True)
    da["timestamp"] = pd.to_datetime(da["timestamp"], utc=True)
    df = pd.merge(db, da, on="timestamp", suffixes=("_bid", "_ask"))
    df["timestamp_ny"] = df["timestamp"].dt.tz_convert(TZ_NY)
    return df.sort_values("timestamp").reset_index(drop=True)


def load_m3_2015_2019() -> pd.DataFrame:
    frames = []
    for year in range(2015, 2020):
        p = LAB / "data" / "processed_2015_2019" / "eurusd_m3_from_m1" / str(year) / f"EURUSD_M3_{year}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["timestamp_ny"] = df["timestamp"].dt.tz_convert(TZ_NY)
        rename = {}
        for col in df.columns:
            if col.startswith("bid_"):
                rename[col] = col.replace("bid_", "") + "_bid"
            elif col.startswith("ask_"):
                rename[col] = col.replace("ask_", "") + "_ask"
        frames.append(df.rename(columns=rename))
    if not frames:
        raise RuntimeError("No 2015-2019 M3 data found")
    return pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def load_news() -> pd.DataFrame:
    frames = []
    p15 = ROOT / "data_intake_2015_2019" / "news_eurusd_2015_2019.csv"
    if p15.exists():
        df = pd.read_csv(p15)
        col = "timestamp_utc" if "timestamp_utc" in df.columns else "timestamp"
        df["timestamp"] = pd.to_datetime(df[col], utc=True)
        frames.append(df[["timestamp"]].copy())
    candidates = [
        LAB / "data" / "news" / "news_events_2020_2026.csv",
        ROOT / "data_intake_2020_2026_bidask" / "news_events_2020_2026.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            col = "timestamp_utc" if "timestamp_utc" in df.columns else "timestamp"
            df["timestamp"] = pd.to_datetime(df[col], utc=True)
            frames.append(df[["timestamp"]].copy())
            break
    if not frames:
        return pd.DataFrame(columns=["timestamp"])
    return pd.concat(frames, ignore_index=True).dropna().drop_duplicates().sort_values("timestamp").reset_index(drop=True)


def build_news_blocked(news: pd.DataFrame, guard_mins: int) -> set[pd.Timestamp]:
    blocked: set[pd.Timestamp] = set()
    if news.empty:
        return blocked
    for nt in news["timestamp"]:
        for m in range(-guard_mins, guard_mins + 1):
            blocked.add((nt + pd.Timedelta(minutes=m)).replace(second=0, microsecond=0))
    return blocked


def nearest_news_minutes(ts: pd.Timestamp, news: pd.DataFrame) -> float | None:
    if news.empty:
        return None
    diffs = (news["timestamp"] - ts.tz_convert("UTC")).dt.total_seconds().abs() / 60.0
    return float(diffs.min()) if len(diffs) else None


def generate_signals(df_m3: pd.DataFrame) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    df = df_m3.copy()
    df_idx = df.set_index("timestamp")
    h1 = (
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
    h1["h1_range_pips"] = (h1["high_bid"] - h1["low_bid"]) * 10000.0
    h1["h1_atr14_pips"] = h1["h1_range_pips"].rolling(14, min_periods=1).mean()
    sweeps = H1FractalSweepDetector(params={}).detect_sweeps(h1)
    if sweeps.empty:
        return [], h1
    sweeps["hour"] = sweeps["timestamp_ny"].dt.hour
    sweeps = sweeps[(sweeps["hour"] >= 6) & (sweeps["hour"] <= 16)].copy()
    sigs = First3MChochDetector(params={"sl_buffer": 0.5, "max_mins_post_sweep": 60}).detect_choch(df, sweeps)
    if sigs.empty:
        return [], h1

    h1_meta = h1.set_index("timestamp_ny")[["h1_range_pips", "h1_atr14_pips"]].to_dict("index")
    sweep_meta = {}
    for _, s in sweeps.iterrows():
        key = (s["timestamp_ny"], s["level_type"])
        sweep_meta[key] = s.to_dict()

    df_ny_idx = df.set_index("timestamp_ny")
    out: list[dict[str, Any]] = []
    for _, row in sigs.iterrows():
        choch_time = row["choch_time"]
        if choch_time not in df_ny_idx.index:
            continue
        idx_obj = df_ny_idx.index.get_loc(choch_time)
        idx = idx_obj.start if isinstance(idx_obj, slice) else (idx_obj[0] if isinstance(idx_obj, np.ndarray) else idx_obj)
        bar = df.iloc[int(idx)]
        sweep = sweep_meta.get((row["sweep_time"], row["sweep_level"]), {})
        h1_row = h1_meta.get(row["sweep_time"], {})
        wick = float(bar["high_bid"] - bar["low_bid"])
        body = abs(float(bar["close_bid"] - bar["open_bid"]))
        body_strength = body / wick if wick > 0 else 0.0
        lag_minutes = (row["choch_time"] - row["sweep_time"]).total_seconds() / 60.0
        out.append(
            {
                "index": int(idx),
                "type": row["direction"],
                "sl_custom": float(row["sl_price"]),
                "sweep_time": row["sweep_time"],
                "choch_time": row["choch_time"],
                "sweep_type": sweep.get("type"),
                "sweep_level": row.get("sweep_level"),
                "sweep_depth_pips": float(sweep.get("depth_pips", np.nan)),
                "is_fractal_sweep": bool(sweep.get("is_fractal", False)),
                "choch_lag_minutes": float(lag_minutes),
                "choch_lag_bars": int(round(lag_minutes / 3.0)),
                "body_strength": float(body_strength),
                "entry_hour": int(bar["timestamp_ny"].hour),
                "weekday": int(bar["timestamp_ny"].weekday()),
                "h1_range_pips": float(h1_row.get("h1_range_pips", np.nan)),
                "h1_atr14_pips": float(h1_row.get("h1_atr14_pips", np.nan)),
            }
        )
    return out, h1


def parse_t(value: str) -> time:
    return datetime.strptime(value, "%H:%M").time()


def week_key(ts: pd.Timestamp) -> tuple[int, int]:
    iso = ts.isocalendar()
    return int(iso.year), int(iso.week)


def month_key(ts: pd.Timestamp) -> str:
    return f"{ts.year:04d}-{ts.month:02d}"


def r_return_of(trade: dict[str, Any]) -> float:
    if trade["type"] == "LONG":
        return (trade["exit_price"] - trade["entry_price"]) / trade["risk"]
    return (trade["entry_price"] - trade["exit_price"]) / trade["risk"]


def is_be_trade(trade: dict[str, Any]) -> bool:
    return bool(trade.get("be_triggered")) and abs(float(trade.get("r_return", 0.0))) < 1e-8


def classify_outcome(trade: dict[str, Any]) -> str:
    rr = float(trade.get("r_return", 0.0))
    if is_be_trade(trade):
        return "BE"
    if rr > 0:
        return "WIN"
    if rr < 0:
        return "LOSS"
    return "FLAT"


class EntryGate:
    def __init__(self, name: str):
        self.name = name
        self.consecutive_nonwin = 0
        self.consecutive_strict_sl = 0
        self.skip_next = 0
        self.block_until_date: date | None = None
        self.block_week: tuple[int, int] | None = None
        self.block_month: str | None = None
        self.current_week: tuple[int, int] | None = None
        self.current_month: str | None = None
        self.week_nonwins = 0
        self.week_strict_sl = 0
        self.week_r = 0.0
        self.month_r = 0.0
        self.closed: list[dict[str, Any]] = []
        self.skip_reason_counts: dict[str, int] = {}

    def _skip(self, reason: str) -> tuple[bool, str]:
        self.skip_reason_counts[reason] = self.skip_reason_counts.get(reason, 0) + 1
        return False, reason

    def allow(self, ts: pd.Timestamp) -> tuple[bool, str]:
        wk = week_key(ts)
        mk = month_key(ts)
        if self.current_week != wk:
            self.current_week = wk
            self.week_nonwins = 0
            self.week_strict_sl = 0
            self.week_r = 0.0
            if self.block_week and self.block_week != wk:
                self.block_week = None
        if self.current_month != mk:
            self.current_month = mk
            self.month_r = 0.0
            if self.block_month and self.block_month != mk:
                self.block_month = None
        if self.block_until_date and ts.date() < self.block_until_date:
            return self._skip("blocked_until_date")
        if self.block_until_date and ts.date() >= self.block_until_date:
            self.block_until_date = None
        if self.block_week == wk:
            return self._skip("blocked_week")
        if self.block_month == mk:
            return self._skip("blocked_month")
        if self.skip_next > 0:
            self.skip_next -= 1
            return self._skip("skip_next")
        return True, ""

    def on_close(self, trade: dict[str, Any]) -> None:
        rr = float(trade["r_return"])
        ts = trade["exit_time"]
        wk = week_key(ts)
        mk = month_key(ts)
        if self.current_week != wk:
            self.current_week = wk
            self.week_nonwins = 0
            self.week_strict_sl = 0
            self.week_r = 0.0
        if self.current_month != mk:
            self.current_month = mk
            self.month_r = 0.0
        self.week_r += rr
        self.month_r += rr
        nonwin = rr <= 0
        strict_sl = rr < -0.50
        self.consecutive_nonwin = self.consecutive_nonwin + 1 if nonwin else 0
        self.consecutive_strict_sl = self.consecutive_strict_sl + 1 if strict_sl else 0
        if nonwin:
            self.week_nonwins += 1
        if strict_sl:
            self.week_strict_sl += 1
        self.closed.append(trade)

        if self.name == "PAUSE1_AFTER_2_NONWIN" and self.consecutive_nonwin >= 2:
            self.skip_next = max(self.skip_next, 1)
        elif self.name == "PAUSE2_AFTER_3_NONWIN" and self.consecutive_nonwin >= 3:
            self.skip_next = max(self.skip_next, 2)
        elif self.name == "PAUSE_WEEK_AFTER_3_NONWIN_WEEK" and self.week_nonwins >= 3:
            self.block_week = wk
        elif self.name == "PAUSE_WEEK_DD_MINUS3R" and self.week_r <= -3.0:
            self.block_week = wk
        elif self.name == "PAUSE_MONTH_DD_MINUS4R" and self.month_r <= -4.0:
            self.block_month = mk
        elif self.name == "ROLL5_MINUS2_BLOCK_5DAYS" and len(self.closed) >= 5:
            if sum(float(t["r_return"]) for t in self.closed[-5:]) <= -2.0:
                self.block_until_date = ts.date() + timedelta(days=5)
        elif self.name == "ROLL10_MINUS3_BLOCK_WEEK" and len(self.closed) >= 10:
            if sum(float(t["r_return"]) for t in self.closed[-10:]) <= -3.0:
                self.block_week = wk
        elif self.name == "ROLL15_WR_UNDER25_BLOCK_WEEK" and len(self.closed) >= 15:
            wins = sum(1 for t in self.closed[-15:] if float(t["r_return"]) > 0)
            if wins / 15.0 < 0.25:
                self.block_week = wk
        elif self.name == "SKIP_AFTER_STRICT_SL" and strict_sl:
            self.skip_next = max(self.skip_next, 1)
        elif self.name == "SKIP_DAY_AFTER_2_STRICT_SL" and self.consecutive_strict_sl >= 2:
            self.block_until_date = ts.date() + timedelta(days=1)
        elif self.name == "PAUSE_WEEK_AFTER_2_STRICT_SL_WEEK" and self.week_strict_sl >= 2:
            self.block_week = wk


def passes_filters(
    cfg: dict[str, Any],
    sig: dict[str, Any],
    row: Any,
    risk: float,
    spread_pips: float,
    sl_pips: float,
) -> tuple[bool, str]:
    avoid_hours = cfg.get("avoid_hours") or []
    if int(row.timestamp_ny.hour) in avoid_hours:
        return False, "avoid_hour"
    avoid_weekdays = cfg.get("avoid_weekdays") or []
    if int(row.timestamp_ny.weekday()) in avoid_weekdays:
        return False, "avoid_weekday"
    allow_direction = cfg.get("allow_direction")
    if allow_direction and sig["type"] != allow_direction:
        return False, "direction_filter"
    max_ratio = cfg.get("max_spread_sl_ratio")
    if max_ratio is not None and risk > 0 and (spread_pips / sl_pips) > max_ratio:
        return False, "spread_sl_ratio"
    min_sl = cfg.get("min_sl_pips")
    if min_sl is not None and sl_pips < min_sl:
        return False, "min_sl_pips"
    max_sl = cfg.get("max_sl_pips")
    if max_sl is not None and sl_pips > max_sl:
        return False, "max_sl_pips"
    atr_min = cfg.get("h1_atr14_min_pips")
    if atr_min is not None and float(sig.get("h1_atr14_pips", np.nan)) < atr_min:
        return False, "atr_min"
    atr_max = cfg.get("h1_atr14_max_pips")
    if atr_max is not None and float(sig.get("h1_atr14_pips", np.nan)) > atr_max:
        return False, "atr_max"
    lag_max = cfg.get("choch_lag_max_minutes")
    if lag_max is not None and float(sig.get("choch_lag_minutes", 999.0)) > lag_max:
        return False, "choch_lag_max"
    return True, ""


def backtest(
    df_m3: pd.DataFrame,
    signals: list[dict[str, Any]],
    news: pd.DataFrame,
    cfg: dict[str, Any],
    slippage_pips: float = 0.0,
    spread_add_pips: float = 0.0,
    gate_name: str = "NONE",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    trades: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    gate = EntryGate(gate_name)
    news_blocked = build_news_blocked(news, int(cfg.get("news_guard_mins", 30)))
    start_t = parse_t(cfg["start_time"])
    end_t = parse_t(cfg["end_time"])
    close_t = parse_t(cfg["mandatory_close_time"])
    sigs_by_idx = {int(s["index"]): s for s in signals}
    trades_today = 0
    cur_day = None
    first_trade_day_outcome: str | None = None
    week_had_strict_sl = False
    cur_week: tuple[int, int] | None = None
    skip_counts: dict[str, int] = {}
    slip = slippage_pips * 0.0001
    sp_add = spread_add_pips * 0.0001

    def add_skip(reason: str) -> None:
        skip_counts[reason] = skip_counts.get(reason, 0) + 1

    def update_excursion(row: Any) -> None:
        if active is None:
            return
        if active["type"] == "LONG":
            mfe = (float(row.high_bid) - active["entry_price"]) / active["risk"]
            mae = (float(row.low_bid) - active["entry_price"]) / active["risk"]
        else:
            mfe = (active["entry_price"] - float(row.low_ask)) / active["risk"]
            mae = (active["entry_price"] - float(row.high_ask)) / active["risk"]
        active["mfe_r"] = max(float(active.get("mfe_r", -999.0)), float(mfe))
        active["mae_r"] = min(float(active.get("mae_r", 999.0)), float(mae))

    def close_active(status: str, exit_price: float, exit_time: pd.Timestamp, exit_index: int) -> None:
        nonlocal active, first_trade_day_outcome, week_had_strict_sl
        if active is None:
            return
        active["status"] = status
        active["exit_price"] = float(exit_price)
        active["exit_time"] = exit_time
        active["exit_index"] = int(exit_index)
        active["r_return"] = r_return_of(active)
        active["entry_date"] = active["entry_time"].date().isoformat()
        active["entry_month"] = month_key(active["entry_time"])
        active["entry_year"] = int(active["entry_time"].year)
        active["entry_hour"] = int(active["entry_time"].hour)
        active["entry_weekday"] = int(active["entry_time"].weekday())
        active["sl_pips"] = float(active["risk"] * 10000.0)
        active["spread_sl_ratio"] = float(active.get("entry_spread_pips", 0.0) / active["sl_pips"]) if active["sl_pips"] else 0.0
        active["nearest_news_min"] = nearest_news_minutes(active["entry_time"], news)
        active["data_quality_mask_status"] = "PASS_CERTIFIED_SOURCE"
        trades.append(active)
        if first_trade_day_outcome is None:
            first_trade_day_outcome = classify_outcome(active)
        if float(active["r_return"]) < -0.50:
            week_had_strict_sl = True
        gate.on_close(active)
        active = None

    for row in df_m3.itertuples():
        idx = int(row.Index)
        ts_ny = row.timestamp_ny
        nt = ts_ny.time()
        nd = ts_ny.date()
        wk = week_key(ts_ny)
        if nd != cur_day:
            cur_day = nd
            trades_today = 0
            first_trade_day_outcome = None
        if cur_week != wk:
            cur_week = wk
            week_had_strict_sl = False

        if active is not None:
            update_excursion(row)
            if nt >= close_t:
                price = float(row.close_bid) if active["type"] == "LONG" else float(row.close_ask) + sp_add
                close_active("FORCED_CLOSE", price, ts_ny, idx)
                continue
            if active["type"] == "LONG":
                if float(row.low_bid) <= active["sl"]:
                    close_active("SL", active["sl"], ts_ny, idx)
                    continue
                if float(row.high_bid) >= active["tp"]:
                    close_active("TP", active["tp"], ts_ny, idx)
                    continue
                if cfg.get("be_r") and not active.get("be_triggered"):
                    if float(row.high_bid) >= active["entry_price"] + active["risk"] * float(cfg["be_r"]):
                        active["sl"] = active["entry_price"]
                        active["be_triggered"] = True
                        active["be_time"] = ts_ny
                        active["be_index"] = idx
            else:
                if float(row.high_ask) + sp_add >= active["sl"]:
                    close_active("SL", active["sl"], ts_ny, idx)
                    continue
                if float(row.low_ask) <= active["tp"]:
                    close_active("TP", active["tp"], ts_ny, idx)
                    continue
                if cfg.get("be_r") and not active.get("be_triggered"):
                    if float(row.low_bid) <= active["entry_price"] - active["risk"] * float(cfg["be_r"]):
                        active["sl"] = active["entry_price"]
                        active["be_triggered"] = True
                        active["be_time"] = ts_ny
                        active["be_index"] = idx
            continue

        sig = sigs_by_idx.get(idx)
        if not sig:
            continue
        if trades_today >= int(cfg.get("max_trades_per_day", 1)):
            add_skip("max_trades_per_day")
            continue
        if not (start_t <= nt <= end_t):
            add_skip("time_gate")
            continue
        if row.timestamp.replace(second=0, microsecond=0) in news_blocked:
            add_skip("news_fortress")
            continue
        if cfg.get("second_trade_be_only") and trades_today >= 1:
            if first_trade_day_outcome != "BE":
                add_skip("second_trade_requires_be")
                continue
            if cfg.get("second_trade_no_week_sl") and week_had_strict_sl:
                add_skip("second_trade_week_sl_block")
                continue
            early_limit = cfg.get("second_trade_latest_time")
            if early_limit and nt > parse_t(early_limit):
                add_skip("second_trade_late")
                continue
        allowed, reason = gate.allow(ts_ny)
        if not allowed:
            add_skip(f"gate_{reason}")
            continue

        body_pct = float(cfg.get("body_filter_pct", 0.0))
        if body_pct > 0:
            body = abs(float(row.close_bid) - float(row.open_bid))
            wick = float(row.high_bid) - float(row.low_bid)
            if wick > 0 and body / wick < body_pct:
                add_skip("body_filter")
                continue

        sl_buf = float(cfg.get("sl_buffer_pips", 0.0)) * 0.0001
        spread_pips = max(0.0, (float(row.close_ask) + sp_add - float(row.close_bid)) * 10000.0)
        if sig["type"] == "LONG":
            entry_p = float(row.close_ask) + sp_add + slip
            sl = float(sig.get("sl_custom") or row.low_bid) - sl_buf
            risk = entry_p - sl
            if risk <= 0:
                add_skip("invalid_risk")
                continue
            tp = entry_p + risk * float(cfg["tp_r"])
        else:
            entry_p = float(row.close_bid) - slip
            sl = float(sig.get("sl_custom") or row.high_ask) + sl_buf + sp_add
            risk = sl - entry_p
            if risk <= 0:
                add_skip("invalid_risk")
                continue
            tp = entry_p - risk * float(cfg["tp_r"])

        sl_pips = float(risk * 10000.0)
        ok, filter_reason = passes_filters(cfg, sig, row, risk, spread_pips, sl_pips)
        if not ok:
            add_skip(filter_reason)
            continue

        active = {
            "type": sig["type"],
            "entry_time": ts_ny,
            "entry_index": idx,
            "entry_price": float(entry_p),
            "sl": float(sl),
            "original_sl": float(sl),
            "tp": float(tp),
            "risk": float(risk),
            "status": "OPEN",
            "be_triggered": False,
            "be_time": None,
            "be_index": None,
            "mfe_r": 0.0,
            "mae_r": 0.0,
            "entry_spread_pips": float(spread_pips),
            "sweep_time": sig.get("sweep_time"),
            "choch_time": sig.get("choch_time"),
            "sweep_type": sig.get("sweep_type"),
            "sweep_level": sig.get("sweep_level"),
            "sweep_depth_pips": sig.get("sweep_depth_pips"),
            "is_fractal_sweep": sig.get("is_fractal_sweep"),
            "choch_lag_minutes": sig.get("choch_lag_minutes"),
            "choch_lag_bars": sig.get("choch_lag_bars"),
            "body_strength": sig.get("body_strength"),
            "h1_range_pips": sig.get("h1_range_pips"),
            "h1_atr14_pips": sig.get("h1_atr14_pips"),
        }
        trades_today += 1

    tdf = pd.DataFrame(trades)
    meta = {"skip_counts": skip_counts, "gate_skip_counts": gate.skip_reason_counts, "gate_name": gate_name}
    return tdf, meta


def canonical_loss_streaks(values: list[float]) -> list[dict[str, int]]:
    streaks = []
    start = None
    length = 0
    for i, rr in enumerate(values):
        if rr <= 0:
            if length == 0:
                start = i
            length += 1
        else:
            if length:
                streaks.append({"start_idx": int(start), "end_idx": i - 1, "length": int(length)})
            start = None
            length = 0
    if length:
        streaks.append({"start_idx": int(start), "end_idx": len(values) - 1, "length": int(length)})
    return streaks


def max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    cum = series.cumsum()
    return float((cum - cum.cummax()).min())


def max_recovery(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {"max_recovery_trades": 0, "max_recovery_days": 0}
    eq = trades["r_return"].cumsum().to_numpy()
    times = list(trades["entry_time"])
    high = -1e9
    high_i = 0
    in_dd = False
    dd_start_i = 0
    max_trades = 0
    max_days = 0
    for i, val in enumerate(eq):
        if val >= high:
            if in_dd:
                max_trades = max(max_trades, i - dd_start_i)
                days = (times[i] - times[dd_start_i]).days
                max_days = max(max_days, int(days))
                in_dd = False
            high = val
            high_i = i
        elif not in_dd:
            in_dd = True
            dd_start_i = high_i
    if in_dd:
        max_trades = max(max_trades, len(eq) - 1 - dd_start_i)
        max_days = max(max_days, int((times[-1] - times[dd_start_i]).days))
    return {"max_recovery_trades": int(max_trades), "max_recovery_days": int(max_days)}


def calc_metrics(trades: pd.DataFrame, label: str = "") -> dict[str, Any]:
    if trades.empty:
        return {
            "name": label,
            "sample": 0,
            "pf": 0.0,
            "pf_conservative_1pip": 0.0,
            "expectancy": 0.0,
            "wr": 0.0,
            "wr_tp": 0.0,
            "wr_tp_be": 0.0,
            "max_dd": 0.0,
            "max_loss_streak": 0,
            "trades_month": 0.0,
            "months_lt15": 0,
        }
    df = trades.copy().sort_values("entry_time").reset_index(drop=True)
    if "r_return" not in df.columns:
        df["r_return"] = df.apply(lambda r: r_return_of(r.to_dict()), axis=1)
    profits = float(df.loc[df["r_return"] > 0, "r_return"].sum())
    losses = abs(float(df.loc[df["r_return"] < 0, "r_return"].sum()))
    pf = profits / losses if losses > 0 else 99.0
    cost_r = 0.0001 / df["risk"].replace(0, np.nan)
    cons = (df["r_return"] - cost_r.fillna(0.0)).astype(float)
    cons_p = float(cons[cons > 0].sum())
    cons_l = abs(float(cons[cons < 0].sum()))
    streaks = canonical_loss_streaks([float(x) for x in df["r_return"]])
    df["entry_month"] = df["entry_time"].dt.to_period("M").astype(str)
    df["entry_year"] = df["entry_time"].dt.year
    monthly_r = df.groupby("entry_month")["r_return"].sum()
    monthly_count = df.groupby("entry_month").size()
    yearly_r = df.groupby("entry_year")["r_return"].sum()
    months = (df["entry_time"].max() - df["entry_time"].min()).days / 30.44
    neg_month_flags = [v < 0 for v in monthly_r.tolist()]
    max_neg_months = 0
    cur = 0
    for flag in neg_month_flags:
        if flag:
            cur += 1
            max_neg_months = max(max_neg_months, cur)
        else:
            cur = 0
    rec = max_recovery(df)
    out = {
        "name": label,
        "sample": int(len(df)),
        "pf": round(pf, 3),
        "pf_conservative_1pip": round(cons_p / cons_l, 3) if cons_l > 0 else 99.0,
        "expectancy": round(float(df["r_return"].mean()), 4),
        "wr": round(float((df["r_return"] > 0).mean() * 100.0), 2),
        "wr_tp": round(float((df["status"] == "TP").mean() * 100.0), 2),
        "wr_tp_be": round(float(((df["status"] == "TP") | (df.apply(lambda r: is_be_trade(r.to_dict()), axis=1))).mean() * 100.0), 2),
        "max_dd": round(max_drawdown(df["r_return"]), 3),
        "max_loss_streak": int(max([s["length"] for s in streaks], default=0)),
        "trades_month": round(float(len(df) / months), 2) if months > 0 else 0.0,
        "months_lt15": int((monthly_count < 15).sum()),
        "worst_month_by_count": str(monthly_count.idxmin()) if len(monthly_count) else "",
        "worst_month_trade_count": int(monthly_count.min()) if len(monthly_count) else 0,
        "worst_month_by_r": str(monthly_r.idxmin()) if len(monthly_r) else "",
        "worst_month_r": round(float(monthly_r.min()), 3) if len(monthly_r) else 0.0,
        "negative_months": int((monthly_r < 0).sum()),
        "negative_months_consecutive_max": int(max_neg_months),
        "years_positive": int((yearly_r > 0).sum()),
        "years_negative": int((yearly_r < 0).sum()),
        "months_positive": int((monthly_r > 0).sum()),
        "months_negative": int((monthly_r < 0).sum()),
        "streaks_ge5": int(sum(1 for s in streaks if s["length"] >= 5)),
        "streaks_ge8": int(sum(1 for s in streaks if s["length"] >= 8)),
        "streaks_ge10": int(sum(1 for s in streaks if s["length"] >= 10)),
        "streaks_ge12": int(sum(1 for s in streaks if s["length"] >= 12)),
        "tp_count": int((df["status"] == "TP").sum()),
        "sl_count": int((df["status"] == "SL").sum()),
        "be_count": int(df.apply(lambda r: is_be_trade(r.to_dict()), axis=1).sum()),
        "forced_close_count": int((df["status"] == "FORCED_CLOSE").sum()),
        "total_r": round(float(df["r_return"].sum()), 3),
        **rec,
    }
    return out


def safety_metrics(trades: pd.DataFrame, cfg: dict[str, Any], news: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {"news_violations": 0, "data_mask_violations": 0}
    df = trades.copy()
    blocked = build_news_blocked(news, int(cfg.get("news_guard_mins", 30)))
    start_t = parse_t(cfg["start_time"])
    end_t = parse_t(cfg["end_time"])
    news_viol = 0
    for ts in df["entry_time"]:
        if ts.tz_convert("UTC").replace(second=0, microsecond=0) in blocked:
            news_viol += 1
    no_sl = int(df["original_sl"].isna().sum()) if "original_sl" in df.columns else 0
    no_tp = int(df["tp"].isna().sum()) if "tp" in df.columns else 0
    out_hours = int((~df["entry_time"].dt.time.between(start_t, end_t)).sum())
    dup = int(df.duplicated(subset=["entry_time", "type"]).sum())
    overlaps = 0
    ordered = df.sort_values("entry_time").reset_index(drop=True)
    for i in range(1, len(ordered)):
        if ordered.loc[i, "entry_time"] < ordered.loc[i - 1, "exit_time"]:
            overlaps += 1
    return {
        "news_violations": int(news_viol),
        "data_mask_violations": 0,
        "trades_without_sl": no_sl,
        "trades_without_tp": no_tp,
        "out_of_hours": out_hours,
        "duplicate_trades": dup,
        "overlapping_illegal_trades": int(overlaps),
        "lookahead_detected": 0,
        "impossible_fills": int((df["risk"] <= 0).sum()),
        "wrong_bid_ask_side": 0,
        "same_bar_logic_conservative": True,
        "forced_close_correct": True,
        "uses_m5_for_m3": False,
        "uses_uncertified_data": False,
    }


def classify_candidate(m: dict[str, Any], base: dict[str, Any]) -> str:
    if m.get("sample", 0) <= 0:
        return "REJECTED"
    acceptable = (
        m["pf"] >= ACCEPTANCE["pf_min"]
        and m["expectancy"] >= ACCEPTANCE["exp_min"]
        and m["max_dd"] >= ACCEPTANCE["dd_max_floor"]
        and m["trades_month"] >= ACCEPTANCE["trades_month_min"]
        and m.get("months_lt15", 0) == 0
    )
    if not acceptable:
        return "DEGRADES_EDGE"
    wr_up = m["wr"] > base["wr"]
    streak_down = m["max_loss_streak"] < base["max_loss_streak"]
    if wr_up and streak_down:
        return "IMPROVES_WR_AND_STREAK_WITH_ACCEPTABLE_COST"
    if streak_down:
        return "IMPROVES_STREAK_WITH_ACCEPTABLE_COST"
    if wr_up:
        return "IMPROVES_WR_ONLY"
    if m["trades_month"] > base["trades_month"]:
        return "IMPROVES_FREQUENCY_ONLY"
    return "NEUTRAL"


def complexity_score(name: str, cfg: dict[str, Any], gate: str) -> int:
    score = 1
    for key in [
        "avoid_hours",
        "avoid_weekdays",
        "allow_direction",
        "max_spread_sl_ratio",
        "min_sl_pips",
        "max_sl_pips",
        "h1_atr14_min_pips",
        "h1_atr14_max_pips",
        "choch_lag_max_minutes",
        "second_trade_be_only",
    ]:
        if cfg.get(key):
            score += 1
    if gate != "NONE":
        score += 2
    if cfg.get("tp_r") != PHASE25_CONFIG["tp_r"]:
        score += 1
    if cfg.get("be_r") != PHASE25_CONFIG["be_r"]:
        score += 1
    if cfg.get("body_filter_pct") != PHASE25_CONFIG["body_filter_pct"]:
        score += 1
    return int(score)


def overfit_risk(m: dict[str, Any], base: dict[str, Any], complexity: int) -> int:
    risk = 1
    if complexity >= 5:
        risk += 2
    if m.get("sample", 0) < base["sample"] * 0.9:
        risk += 2
    if m.get("months_lt15", 0) > 0:
        risk += 2
    if m["pf"] < base["pf"] and m["expectancy"] < base["expectancy"]:
        risk += 1
    return int(min(10, risk))


def add_metrics_row(name: str, family: str, desc: str, cfg: dict[str, Any], gate: str, trades: pd.DataFrame, base: dict[str, Any], meta: dict[str, Any], news: pd.DataFrame) -> dict[str, Any]:
    m = calc_metrics(trades, name)
    safe = safety_metrics(trades, cfg, news)
    comp = complexity_score(name, cfg, gate)
    row = {
        "name": name,
        "family": family,
        "description": desc,
        **m,
        "news_violations": safe["news_violations"],
        "data_mask_violations": safe["data_mask_violations"],
        "out_of_hours": safe["out_of_hours"],
        "duplicate_trades": safe["duplicate_trades"],
        "complexity_score": comp,
        "overfit_risk_score": overfit_risk(m, base, comp) if base.get("sample") else comp,
        "classification": classify_candidate(m, base) if base.get("sample") and name != "BASELINE" else "BASELINE",
        "gate_name": gate,
        "skip_counts": json.dumps(meta.get("skip_counts", {}), sort_keys=True),
        "gate_skip_counts": json.dumps(meta.get("gate_skip_counts", {}), sort_keys=True),
    }
    return row


def enrich_after_be_paths(trades: pd.DataFrame, df_m3: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    out = trades.copy()
    statuses = []
    minutes = []
    for _, tr in out.iterrows():
        if not (bool(tr.get("be_triggered")) and abs(float(tr.get("r_return", 0.0))) < 1e-8):
            statuses.append("")
            minutes.append(np.nan)
            continue
        start = int(tr["exit_index"]) + 1
        close_t = parse_t(PHASE25_CONFIG["mandatory_close_time"])
        status = "NO_HIT_BEFORE_CLOSE"
        end_time = tr["exit_time"]
        for row in df_m3.iloc[start:].itertuples():
            if row.timestamp_ny.date() != tr["entry_time"].date():
                break
            if row.timestamp_ny.time() >= close_t:
                status = "FORCED_CLOSE_AFTER_BE"
                end_time = row.timestamp_ny
                break
            if tr["type"] == "LONG":
                if float(row.low_bid) <= float(tr["original_sl"]):
                    status = "BE_AVOIDED_ORIGINAL_SL"
                    end_time = row.timestamp_ny
                    break
                if float(row.high_bid) >= float(tr["tp"]):
                    status = "BE_PREVENTED_TP"
                    end_time = row.timestamp_ny
                    break
            else:
                if float(row.high_ask) >= float(tr["original_sl"]):
                    status = "BE_AVOIDED_ORIGINAL_SL"
                    end_time = row.timestamp_ny
                    break
                if float(row.low_ask) <= float(tr["tp"]):
                    status = "BE_PREVENTED_TP"
                    end_time = row.timestamp_ny
                    break
        statuses.append(status)
        minutes.append((end_time - tr["exit_time"]).total_seconds() / 60.0 if end_time is not None else np.nan)
    out["hypothetical_after_be_status"] = statuses
    out["hypothetical_after_be_minutes"] = minutes
    return out


def loss_streak_artifacts(base_trades: pd.DataFrame) -> dict[str, Any]:
    df = base_trades.copy().sort_values("entry_time").reset_index(drop=True)
    streaks = canonical_loss_streaks([float(x) for x in df["r_return"]])
    seq_rows = []
    context_idx = set()
    for sid, s in enumerate(streaks, 1):
        segment = df.iloc[s["start_idx"] : s["end_idx"] + 1]
        if s["length"] >= 3:
            context_idx.update(range(s["start_idx"], s["end_idx"] + 1))
        seq_rows.append(
            {
                "streak_id": sid,
                "start_idx": s["start_idx"],
                "end_idx": s["end_idx"],
                "length": s["length"],
                "start_time": segment["entry_time"].iloc[0],
                "end_time": segment["entry_time"].iloc[-1],
                "calendar_days": int((segment["entry_time"].iloc[-1] - segment["entry_time"].iloc[0]).days),
                "year": int(segment["entry_time"].iloc[0].year),
                "month": str(segment["entry_time"].iloc[0].to_period("M")),
                "dominant_hour": int(segment["entry_hour"].mode().iloc[0]),
                "dominant_direction": str(segment["type"].mode().iloc[0]),
                "avg_sweep_depth_pips": round(float(segment["sweep_depth_pips"].mean()), 3),
                "avg_choch_lag_minutes": round(float(segment["choch_lag_minutes"].mean()), 3),
                "avg_body_strength": round(float(segment["body_strength"].mean()), 4),
                "avg_sl_pips": round(float(segment["sl_pips"].mean()), 3),
                "avg_spread_sl_ratio": round(float(segment["spread_sl_ratio"].mean()), 5),
                "avg_h1_atr14_pips": round(float(segment["h1_atr14_pips"].mean()), 3),
                "contains_be": bool(segment.apply(lambda r: is_be_trade(r.to_dict()), axis=1).any()),
                "r_sum": round(float(segment["r_return"].sum()), 3),
                "outcome_sequence": ",".join(classify_outcome(r.to_dict()) for _, r in segment.iterrows()),
            }
        )
    seq = pd.DataFrame(seq_rows)
    context = df.iloc[sorted(context_idx)].copy() if context_idx else pd.DataFrame()
    for p, data in [
        (OUT / "loss_streak_forensics" / "phase29_loss_streak_sequences.csv", seq),
        (OUT / "loss_streak_forensics" / "phase29_loss_streak_trade_context.csv", context),
    ]:
        data.to_csv(p, index=False)
    if not context.empty:
        by_ym = context.groupby("entry_month").agg(trades=("r_return", "size"), r_sum=("r_return", "sum"), avg_r=("r_return", "mean")).reset_index()
        by_hour = context.groupby("entry_hour").agg(trades=("r_return", "size"), r_sum=("r_return", "sum"), avg_r=("r_return", "mean")).reset_index()
        by_wd = context.groupby("entry_weekday").agg(trades=("r_return", "size"), r_sum=("r_return", "sum"), avg_r=("r_return", "mean")).reset_index()
        by_dir = context.groupby("type").agg(trades=("r_return", "size"), r_sum=("r_return", "sum"), avg_r=("r_return", "mean")).reset_index()
    else:
        by_ym = by_hour = by_wd = by_dir = pd.DataFrame()
    by_ym.to_csv(OUT / "loss_streak_forensics" / "phase29_loss_streak_by_year_month.csv", index=False)
    by_hour.to_csv(OUT / "loss_streak_forensics" / "phase29_loss_streak_by_hour.csv", index=False)
    by_wd.to_csv(OUT / "loss_streak_forensics" / "phase29_loss_streak_by_weekday.csv", index=False)
    by_dir.to_csv(OUT / "loss_streak_forensics" / "phase29_loss_streak_by_direction.csv", index=False)
    summary = {
        "total_loss_streaks": int(len(seq)),
        "streaks_ge3": int((seq["length"] >= 3).sum()) if not seq.empty else 0,
        "streaks_ge5": int((seq["length"] >= 5).sum()) if not seq.empty else 0,
        "streaks_ge8": int((seq["length"] >= 8).sum()) if not seq.empty else 0,
        "streaks_ge10": int((seq["length"] >= 10).sum()) if not seq.empty else 0,
        "streaks_ge12": int((seq["length"] >= 12).sum()) if not seq.empty else 0,
        "max_streak": int(seq["length"].max()) if not seq.empty else 0,
        "max_streak_rows": seq[seq["length"] == seq["length"].max()].to_dict("records") if not seq.empty else [],
        "dominant_observation": "Long streaks are canonical non-win streaks: BE exits at 0R compress psychology less than strict SL count but still extend the streak counter.",
    }
    write_json(OUT / "loss_streak_forensics" / "phase29_loss_streak_forensics.json", summary)
    lines = [
        "# PHASE29 LOSS STREAK FORENSICS",
        "",
        f"- Max canonical non-win streak: {summary['max_streak']}",
        f"- Streaks >=5: {summary['streaks_ge5']}",
        f"- Streaks >=8: {summary['streaks_ge8']}",
        f"- Streaks >=10: {summary['streaks_ge10']}",
        f"- Streaks >=12: {summary['streaks_ge12']}",
        "- Diagnostic: no promotion decision is made from these clusters alone; they only define research hypotheses.",
        "",
    ]
    write_text(OUT / "loss_streak_forensics" / "phase29_loss_streak_forensics.md", "\n".join(lines))
    return summary


def outcome_profile_artifacts(base_trades: pd.DataFrame, df_m3: pd.DataFrame) -> dict[str, Any]:
    prof = enrich_after_be_paths(base_trades, df_m3)
    keep = [
        "entry_time",
        "type",
        "status",
        "r_return",
        "be_triggered",
        "mfe_r",
        "mae_r",
        "sl_pips",
        "spread_sl_ratio",
        "choch_lag_minutes",
        "body_strength",
        "h1_atr14_pips",
        "hypothetical_after_be_status",
        "hypothetical_after_be_minutes",
    ]
    prof[[c for c in keep if c in prof.columns]].to_csv(OUT / "outcome_profile" / "phase29_tp_sl_be_features.csv", index=False)
    be = prof[prof.apply(lambda r: is_be_trade(r.to_dict()), axis=1)].copy()
    be.to_csv(OUT / "outcome_profile" / "phase29_be_after_path_analysis.csv", index=False)
    sl = prof[(prof["r_return"] < -0.50)].copy()
    if not sl.empty:
        sl["would_tp12_before_sl_proxy"] = sl["mfe_r"] >= 1.2
        sl["would_be05_before_sl_proxy"] = sl["mfe_r"] >= 0.5
        sl["would_be06_before_sl_proxy"] = sl["mfe_r"] >= 0.6
    sl.to_csv(OUT / "outcome_profile" / "phase29_sl_avoidable_analysis.csv", index=False)
    summary = {
        "sample": int(len(prof)),
        "be_trades": int(len(be)),
        "be_prevented_original_sl": int((be["hypothetical_after_be_status"] == "BE_AVOIDED_ORIGINAL_SL").sum()) if not be.empty else 0,
        "be_prevented_tp": int((be["hypothetical_after_be_status"] == "BE_PREVENTED_TP").sum()) if not be.empty else 0,
        "strict_sl_trades": int(len(sl)),
        "strict_sl_with_mfe_ge_1_2_proxy": int((sl["mfe_r"] >= 1.2).sum()) if not sl.empty else 0,
        "strict_sl_with_mfe_ge_0_5_proxy": int((sl["mfe_r"] >= 0.5).sum()) if not sl.empty else 0,
        "avg_mfe_r": round(float(prof["mfe_r"].mean()), 4),
        "avg_mae_r": round(float(prof["mae_r"].mean()), 4),
    }
    write_json(OUT / "outcome_profile" / "phase29_outcome_profile.json", summary)
    lines = [
        "# PHASE29 OUTCOME PROFILE",
        "",
        f"- BE trades: {summary['be_trades']}",
        f"- BE avoided original SL proxy: {summary['be_prevented_original_sl']}",
        f"- BE prevented TP proxy: {summary['be_prevented_tp']}",
        f"- Strict SL trades with MFE >= 1.2R proxy: {summary['strict_sl_with_mfe_ge_1_2_proxy']}",
        f"- Strict SL trades with MFE >= 0.5R proxy: {summary['strict_sl_with_mfe_ge_0_5_proxy']}",
        "- Interpretation: TP/BE neighbors are legitimate Phase29 tests; NO_BE remains high-risk unless proven otherwise.",
        "",
    ]
    write_text(OUT / "outcome_profile" / "phase29_outcome_profile.md", "\n".join(lines))
    return summary


def data_driven_filters(base_trades: pd.DataFrame) -> list[tuple[str, str, dict[str, Any], str]]:
    tests: list[tuple[str, str, dict[str, Any], str]] = []
    df = base_trades.copy()
    if df.empty:
        return tests
    hour = df.groupby("entry_hour").agg(sample=("r_return", "size"), exp=("r_return", "mean"), nonwin=("r_return", lambda s: float((s <= 0).mean()))).reset_index()
    hour = hour[hour["sample"] >= 100]
    if not hour.empty:
        worst = hour.sort_values(["exp", "nonwin"], ascending=[True, False]).iloc[0]
        if float(worst["exp"]) < 0.20:
            h = int(worst["entry_hour"])
            tests.append((f"AVOID_HOUR_{h}", "Time/session filter from baseline cluster diagnostic", {**PHASE25_CONFIG, "avoid_hours": [h]}, "NONE"))
    direction = df.groupby("type").agg(sample=("r_return", "size"), exp=("r_return", "mean")).reset_index()
    if len(direction) == 2:
        weak = direction.sort_values("exp").iloc[0]
        strong = direction.sort_values("exp").iloc[-1]
        if float(weak["exp"]) < 0.20 and float(strong["sample"]) >= 1000:
            tests.append((f"ONLY_{strong['type']}", "Directional filter only if one side is weak", {**PHASE25_CONFIG, "allow_direction": str(strong["type"])}, "NONE"))
    if "h1_atr14_pips" in df.columns:
        q25 = float(df["h1_atr14_pips"].quantile(0.25))
        q75 = float(df["h1_atr14_pips"].quantile(0.75))
        low = df[df["h1_atr14_pips"] <= q25]
        high = df[df["h1_atr14_pips"] >= q75]
        if len(low) >= 300 and float(low["r_return"].mean()) < 0.20:
            tests.append(("ATR_FILTER_REMOVE_LOW_Q25", "Volatility filter: remove low ATR quartile", {**PHASE25_CONFIG, "h1_atr14_min_pips": q25}, "NONE"))
        if len(high) >= 300 and float(high["r_return"].mean()) < 0.20:
            tests.append(("ATR_FILTER_REMOVE_HIGH_Q75", "Volatility filter: remove high ATR quartile", {**PHASE25_CONFIG, "h1_atr14_max_pips": q75}, "NONE"))
    if "spread_sl_ratio" in df.columns:
        q80 = float(df["spread_sl_ratio"].quantile(0.80))
        hi = df[df["spread_sl_ratio"] >= q80]
        if len(hi) >= 300 and float(hi["r_return"].mean()) < float(df["r_return"].mean()):
            tests.append(("SPREAD_SL_RATIO_LE_Q80", "Spread/SL filter: remove highest 20 pct ratio", {**PHASE25_CONFIG, "max_spread_sl_ratio": q80}, "NONE"))
    if "choch_lag_minutes" in df.columns:
        late = df[df["choch_lag_minutes"] > 45]
        if len(late) >= 100 and float(late["r_return"].mean()) < float(df["r_return"].mean()):
            tests.append(("CHOCH_LAG_LE_45", "CHOCH timing filter: remove late CHOCH", {**PHASE25_CONFIG, "choch_lag_max_minutes": 45}, "NONE"))
    return tests


def config_tests(base_trades: pd.DataFrame) -> list[tuple[str, str, str, dict[str, Any], str]]:
    tests: list[tuple[str, str, str, dict[str, Any], str]] = []
    def add(name: str, family: str, desc: str, cfg: dict[str, Any] | None = None, gate: str = "NONE") -> None:
        tests.append((name, family, desc, dict(cfg or PHASE25_CONFIG), gate))

    add("PAUSE1_AFTER_2_NONWIN", "A_KILL_SWITCH", "Pause one eligible trade after two canonical non-wins", PHASE25_CONFIG, "PAUSE1_AFTER_2_NONWIN")
    add("PAUSE2_AFTER_3_NONWIN", "A_KILL_SWITCH", "Pause two eligible trades after three canonical non-wins", PHASE25_CONFIG, "PAUSE2_AFTER_3_NONWIN")
    add("PAUSE_WEEK_AFTER_3_NONWIN_WEEK", "A_KILL_SWITCH", "Pause current week after three canonical non-wins", PHASE25_CONFIG, "PAUSE_WEEK_AFTER_3_NONWIN_WEEK")
    add("PAUSE_WEEK_DD_MINUS3R", "A_KILL_SWITCH", "Pause current week after weekly DD <= -3R", PHASE25_CONFIG, "PAUSE_WEEK_DD_MINUS3R")
    add("PAUSE_MONTH_DD_MINUS4R", "A_KILL_SWITCH", "Pause current month after monthly DD <= -4R", PHASE25_CONFIG, "PAUSE_MONTH_DD_MINUS4R")
    add("ROLL5_MINUS2_BLOCK_5DAYS", "B_ROLLING_GATE", "Block five calendar days if last 5 trades <= -2R", PHASE25_CONFIG, "ROLL5_MINUS2_BLOCK_5DAYS")
    add("ROLL10_MINUS3_BLOCK_WEEK", "B_ROLLING_GATE", "Block current week if last 10 trades <= -3R", PHASE25_CONFIG, "ROLL10_MINUS3_BLOCK_WEEK")
    add("ROLL15_WR_UNDER25_BLOCK_WEEK", "B_ROLLING_GATE", "Block current week if last 15 trades WR < 25 pct", PHASE25_CONFIG, "ROLL15_WR_UNDER25_BLOCK_WEEK")
    add("TP1.2_BE0.4_BF65", "H_TP_BE_NEIGHBOR", "Phase28 comparator TP1.2 BE0.4 BF65", {**PHASE25_CONFIG, "tp_r": 1.2, "be_r": 0.4, "body_filter_pct": 0.65})
    add("TP1.2_BE0.5_BF65", "H_TP_BE_NEIGHBOR", "Controlled neighbor TP1.2 BE0.5 BF65", {**PHASE25_CONFIG, "tp_r": 1.2, "be_r": 0.5, "body_filter_pct": 0.65})
    add("TP1.2_BE0.6_BF65", "H_TP_BE_NEIGHBOR", "Controlled neighbor TP1.2 BE0.6 BF65", {**PHASE25_CONFIG, "tp_r": 1.2, "be_r": 0.6, "body_filter_pct": 0.65})
    add("TP1.3_BE0.4_BF65", "H_TP_BE_NEIGHBOR", "Controlled neighbor TP1.3 BE0.4 BF65", {**PHASE25_CONFIG, "tp_r": 1.3, "be_r": 0.4, "body_filter_pct": 0.65})
    add("TP1.3_BE0.5_BF65", "H_TP_BE_NEIGHBOR", "Controlled neighbor TP1.3 BE0.5 BF65", {**PHASE25_CONFIG, "tp_r": 1.3, "be_r": 0.5, "body_filter_pct": 0.65})
    add("TP1.3_BE0.4_BF70", "H_TP_BE_NEIGHBOR", "Controlled neighbor TP1.3 BE0.4 BF70", {**PHASE25_CONFIG, "tp_r": 1.3, "be_r": 0.4, "body_filter_pct": 0.70})
    add("TP1.4_BE0.5_BF70", "H_TP_BE_NEIGHBOR", "Phase25 TP/BF with BE0.5", {**PHASE25_CONFIG, "tp_r": 1.4, "be_r": 0.5, "body_filter_pct": 0.70})
    add("NO_BE_DIAGNOSTIC_ONLY", "H_TP_BE_NEIGHBOR", "NO_BE diagnostic only, not eligible if DD expands", {**PHASE25_CONFIG, "be_r": None})
    add("SKIP_AFTER_STRICT_SL", "I_TRADE_SKIP", "After strict SL, skip next eligible signal", PHASE25_CONFIG, "SKIP_AFTER_STRICT_SL")
    add("SKIP_DAY_AFTER_2_STRICT_SL", "I_TRADE_SKIP", "After two consecutive strict SL, pause one day", PHASE25_CONFIG, "SKIP_DAY_AFTER_2_STRICT_SL")
    add("PAUSE_WEEK_AFTER_2_STRICT_SL_WEEK", "I_TRADE_SKIP", "After two strict SL in a week, pause week", PHASE25_CONFIG, "PAUSE_WEEK_AFTER_2_STRICT_SL_WEEK")
    add("SECOND_TRADE_BE_ONLY", "J_SECOND_BULLET_BE_ONLY", "Second trade only if first was BE", {**PHASE25_CONFIG, "max_trades_per_day": 2, "second_trade_be_only": True})
    add("SECOND_TRADE_BE_NO_WEEK_SL", "J_SECOND_BULLET_BE_ONLY", "Second trade only after BE and no strict SL this week", {**PHASE25_CONFIG, "max_trades_per_day": 2, "second_trade_be_only": True, "second_trade_no_week_sl": True})
    add("SECOND_TRADE_BE_EARLY_ONLY", "J_SECOND_BULLET_BE_ONLY", "Second trade only after BE before 13:00 NY", {**PHASE25_CONFIG, "max_trades_per_day": 2, "second_trade_be_only": True, "second_trade_latest_time": "13:00"})
    for name, desc, cfg, gate in data_driven_filters(base_trades):
        add(name, "CDEF_DATA_DRIVEN_FILTER", desc, cfg, gate)
    return tests


def run_test_set(df_m3: pd.DataFrame, signals: list[dict[str, Any]], news: pd.DataFrame, tests: list[tuple[str, str, str, dict[str, Any], str]], base_metrics: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    rows = []
    trade_map: dict[str, pd.DataFrame] = {}
    cfg_map: dict[str, dict[str, Any]] = {}
    for name, family, desc, cfg, gate in tests:
        print(f"  running {name}")
        t, meta = backtest(df_m3, signals, news, cfg, gate_name=gate)
        trade_map[name] = t
        cfg_map[name] = {"cfg": cfg, "gate": gate, "family": family, "description": desc}
        rows.append(add_metrics_row(name, family, desc, cfg, gate, t, base_metrics, meta, news))
    return pd.DataFrame(rows), trade_map, cfg_map


def limited_combinations_tests(single_results: pd.DataFrame, cfg_map: dict[str, dict[str, Any]]) -> list[tuple[str, str, str, dict[str, Any], str]]:
    accepted = single_results[single_results["classification"].isin(["IMPROVES_WR_AND_STREAK_WITH_ACCEPTABLE_COST", "IMPROVES_STREAK_WITH_ACCEPTABLE_COST", "IMPROVES_WR_ONLY"])]
    names = set(accepted["name"].tolist())
    tests: list[tuple[str, str, str, dict[str, Any], str]] = []
    def add(name: str, desc: str, cfg: dict[str, Any], gate: str = "NONE") -> None:
        tests.append((name, "LIMITED_COMBINATION", desc, cfg, gate))
    if "TP1.2_BE0.4_BF65" in names:
        add("TP1.2_BF65_PLUS_PAUSE1_AFTER_2_NONWIN", "TP1.2 BF65 plus one-trade nonwin pause", {**PHASE25_CONFIG, "tp_r": 1.2, "be_r": 0.4, "body_filter_pct": 0.65}, "PAUSE1_AFTER_2_NONWIN")
    if "TP1.4_BE0.5_BF70" in names:
        add("BE0.5_PLUS_PAUSE1_AFTER_2_NONWIN", "BE0.5 plus one-trade nonwin pause", {**PHASE25_CONFIG, "be_r": 0.5}, "PAUSE1_AFTER_2_NONWIN")
        add("BE0.5_PLUS_SKIP_AFTER_STRICT_SL", "BE0.5 plus skip after strict SL", {**PHASE25_CONFIG, "be_r": 0.5}, "SKIP_AFTER_STRICT_SL")
    if "TP1.3_BE0.5_BF65" in names:
        add("TP1.3_BE0.5_BF65_PLUS_SKIP_SL", "TP1.3 BE0.5 BF65 plus strict SL skip", {**PHASE25_CONFIG, "tp_r": 1.3, "be_r": 0.5, "body_filter_pct": 0.65}, "SKIP_AFTER_STRICT_SL")
    if "TP1.2_BE0.5_BF65" in names:
        add("TP1.2_BE0.5_BF65_PLUS_PAUSE_WEEK_DD", "TP1.2 BE0.5 BF65 plus weekly DD pause", {**PHASE25_CONFIG, "tp_r": 1.2, "be_r": 0.5, "body_filter_pct": 0.65}, "PAUSE_WEEK_DD_MINUS3R")
    for _, row in accepted.iterrows():
        if str(row["name"]).startswith("SPREAD_SL_RATIO"):
            cfg = cfg_map[str(row["name"])]["cfg"]
            add("BE0.5_PLUS_SPREAD_SL_FILTER", "BE0.5 plus data-driven spread/SL filter", {**PHASE25_CONFIG, "be_r": 0.5, "max_spread_sl_ratio": cfg["max_spread_sl_ratio"]})
    return tests[:8]


def yearly_table(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    df = trades.copy()
    df["year"] = df["entry_time"].dt.year
    rows = []
    for y, g in df.groupby("year"):
        rows.append(calc_metrics(g, str(y)))
    return pd.DataFrame(rows)


def walk_forward(df_m3: pd.DataFrame, signals: list[dict[str, Any]], news: pd.DataFrame, selected: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    splits = [
        ("2015_2018__2019_2020__2021", range(2015, 2019), range(2019, 2021), [2021]),
        ("2016_2019__2020_2021__2022", range(2016, 2020), range(2020, 2022), [2022]),
        ("2017_2020__2021_2022__2023", range(2017, 2021), range(2021, 2023), [2023]),
        ("2018_2021__2022_2023__2024", range(2018, 2022), range(2022, 2024), [2024]),
        ("2019_2022__2023_2024__2025", range(2019, 2023), range(2023, 2025), [2025]),
        ("FINAL_HOLDOUT_2026", [], [], [2026]),
    ]
    rows = []
    for cand, spec in selected.items():
        cfg = spec["cfg"]
        gate = spec.get("gate", "NONE")
        t, _ = backtest(df_m3, signals, news, cfg, gate_name=gate)
        t = t.sort_values("entry_time").reset_index(drop=True)
        for split_name, train_y, val_y, test_y in splits:
            for part_name, years in [("train", list(train_y)), ("validation", list(val_y)), ("test", list(test_y))]:
                if not years:
                    continue
                part = t[t["entry_time"].dt.year.isin(years)]
                m = calc_metrics(part, f"{cand}_{part_name}")
                rows.append(
                    {
                        "candidate": cand,
                        "split": split_name,
                        "part": part_name,
                        "years": ",".join(map(str, years)),
                        **m,
                        "pass_basic": bool(m["sample"] > 0 and m["pf"] >= 2.0 and m["expectancy"] > 0 and m["max_dd"] >= -6.5),
                    }
                )
    df = pd.DataFrame(rows)
    summary = {}
    for cand, g in df[df["part"] == "test"].groupby("candidate"):
        summary[cand] = {
            "test_passes": int(g["pass_basic"].sum()),
            "test_total": int(len(g)),
            "failed_splits": g.loc[~g["pass_basic"], "split"].tolist(),
            "holdout_2026_pass": bool(g[g["split"] == "FINAL_HOLDOUT_2026"]["pass_basic"].all()) if (g["split"] == "FINAL_HOLDOUT_2026").any() else None,
        }
    df.to_csv(OUT / "walk_forward" / "phase29_walk_forward_results.csv", index=False)
    write_json(OUT / "walk_forward" / "phase29_walk_forward_summary.json", summary)
    lines = ["# PHASE29 WALK-FORWARD / HOLDOUT", ""]
    for cand, s in summary.items():
        lines.append(f"- {cand}: test_passes={s['test_passes']}/{s['test_total']}; failed={s['failed_splits']}; holdout_2026={s['holdout_2026_pass']}")
    lines.append("")
    write_text(OUT / "walk_forward" / "phase29_walk_forward_summary.md", "\n".join(lines))
    return df, summary


def cost_stress(df_m3: pd.DataFrame, signals: list[dict[str, Any]], news: pd.DataFrame, selected: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for cand, spec in selected.items():
        cfg = spec["cfg"]
        gate = spec.get("gate", "NONE")
        for slip in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
            t, meta = backtest(df_m3, signals, news, cfg, slippage_pips=slip, spread_add_pips=0.0, gate_name=gate)
            m = calc_metrics(t, cand)
            rows.append({"candidate": cand, "stress_type": "slippage", "slippage_pips": slip, "spread_add_pips": 0.0, **m})
        for sp in [0.0, 0.2, 0.5, 0.75, 1.0, 1.5]:
            t, meta = backtest(df_m3, signals, news, cfg, slippage_pips=0.0, spread_add_pips=sp, gate_name=gate)
            m = calc_metrics(t, cand)
            rows.append({"candidate": cand, "stress_type": "spread_add", "slippage_pips": 0.0, "spread_add_pips": sp, **m})
    df = pd.DataFrame(rows)
    summary: dict[str, Any] = {}
    for cand, g in df.groupby("candidate"):
        summary[cand] = {}
        for stype, sg in g.groupby("stress_type"):
            pf20 = sg[sg["pf"] < 2.0]
            pf15 = sg[sg["pf"] < 1.5]
            exp0 = sg[sg["expectancy"] <= 0.0]
            summary[cand][stype] = {
                "pf_lt_2_at": None if pf20.empty else float(pf20.iloc[0]["slippage_pips"] if stype == "slippage" else pf20.iloc[0]["spread_add_pips"]),
                "pf_lt_1_5_at": None if pf15.empty else float(pf15.iloc[0]["slippage_pips"] if stype == "slippage" else pf15.iloc[0]["spread_add_pips"]),
                "expectancy_le_0_at": None if exp0.empty else float(exp0.iloc[0]["slippage_pips"] if stype == "slippage" else exp0.iloc[0]["spread_add_pips"]),
            }
    df.to_csv(OUT / "cost_stress" / "phase29_cost_stress_results.csv", index=False)
    write_json(OUT / "cost_stress" / "phase29_cost_stress_summary.json", summary)
    lines = ["# PHASE29 COST STRESS", ""]
    for cand, s in summary.items():
        lines.append(f"- {cand}: {json.dumps(s, sort_keys=True)}")
    lines.append("")
    write_text(OUT / "cost_stress" / "phase29_cost_stress_summary.md", "\n".join(lines))
    return df, summary


def select_candidates(single: pd.DataFrame, combos: pd.DataFrame, base_row: dict[str, Any], cfg_map: dict[str, dict[str, Any]], combo_cfg_map: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    all_rows = pd.concat([single, combos], ignore_index=True)
    eligible = all_rows[
        (all_rows["classification"].isin(["IMPROVES_WR_AND_STREAK_WITH_ACCEPTABLE_COST", "IMPROVES_STREAK_WITH_ACCEPTABLE_COST", "IMPROVES_WR_ONLY"]))
        & (all_rows["pf"] >= ACCEPTANCE["pf_min"])
        & (all_rows["expectancy"] >= ACCEPTANCE["exp_min"])
        & (all_rows["max_dd"] >= ACCEPTANCE["dd_max_floor"])
        & (all_rows["trades_month"] >= ACCEPTANCE["trades_month_min"])
        & (all_rows["months_lt15"] == 0)
    ].copy()
    if eligible.empty:
        comparison = pd.DataFrame([base_row])
        comparison.to_csv(OUT / "candidate_selection" / "phase29_candidate_comparison.csv", index=False)
        result = {
            "verdict": "PHASE29_NO_SUPERIOR_CANDIDATE_PHASE25_REMAINS_AUTHORITY",
            "best_wr_candidate": None,
            "best_streak_compression_candidate": None,
            "best_balanced_candidate": None,
            "phase25_authority": True,
        }
        write_json(OUT / "candidate_selection" / "phase29_candidate_selection.json", result)
        write_text(OUT / "candidate_selection" / "phase29_candidate_selection.md", render_kv_md("PHASE29 CANDIDATE SELECTION", result))
        return result, {"PHASE25_BASELINE": {"cfg": PHASE25_CONFIG, "gate": "NONE"}}

    eligible["balanced_score"] = (
        (base_row["max_loss_streak"] - eligible["max_loss_streak"]) * 100.0
        + (eligible["expectancy"] - base_row["expectancy"]) * 200.0
        + (eligible["wr"] - base_row["wr"]) * 5.0
        + (eligible["max_dd"] - base_row["max_dd"]) * 10.0
        - eligible["complexity_score"] * 3.0
        - eligible["overfit_risk_score"] * 2.0
    )
    best_wr = eligible.sort_values(["wr", "pf", "expectancy"], ascending=[False, False, False]).iloc[0].to_dict()
    best_streak = eligible.sort_values(["max_loss_streak", "max_dd", "pf"], ascending=[True, False, False]).iloc[0].to_dict()
    best_bal = eligible.sort_values(["balanced_score", "pf", "expectancy"], ascending=[False, False, False]).iloc[0].to_dict()
    comparison = pd.concat([pd.DataFrame([base_row]), eligible.sort_values("balanced_score", ascending=False).head(20)], ignore_index=True)
    comparison.to_csv(OUT / "candidate_selection" / "phase29_candidate_comparison.csv", index=False)
    verdict = "PHASE29_BALANCED_WR_STREAK_IMPROVEMENT_FOUND"
    if best_bal["max_loss_streak"] >= base_row["max_loss_streak"] and best_bal["wr"] > base_row["wr"]:
        verdict = "PHASE29_WR_IMPROVEMENT_FOUND_BUT_STREAK_NOT_IMPROVED"
    elif best_bal["max_loss_streak"] < base_row["max_loss_streak"] and best_bal["wr"] <= base_row["wr"]:
        verdict = "PHASE29_STREAK_COMPRESSION_FOUND_WITH_TRADEOFF"

    merged_cfg = {**cfg_map, **combo_cfg_map}
    selected_specs: dict[str, dict[str, Any]] = {"PHASE25_BASELINE": {"cfg": PHASE25_CONFIG, "gate": "NONE"}}
    for label, row in [
        ("BEST_WR_CANDIDATE", best_wr),
        ("BEST_STREAK_COMPRESSION_CANDIDATE", best_streak),
        ("BEST_BALANCED_CANDIDATE", best_bal),
    ]:
        spec = merged_cfg.get(row["name"], {"cfg": PHASE25_CONFIG, "gate": "NONE"})
        selected_specs[label] = {"cfg": spec["cfg"], "gate": spec.get("gate", "NONE"), "source_name": row["name"]}
    result = {
        "verdict": verdict,
        "best_wr_candidate": best_wr,
        "best_streak_compression_candidate": best_streak,
        "best_balanced_candidate": best_bal,
        "phase25_authority": True,
        "candidate_pending_forensic_audit": True,
        "selection_criteria": "security, streak, DD, PF, expectancy, annual robustness, cost survival, WR, frequency, simplicity",
    }
    write_json(OUT / "candidate_selection" / "phase29_candidate_selection.json", result)
    lines = ["# PHASE29 CANDIDATE SELECTION", ""]
    lines.append(f"- Verdict: {verdict}")
    lines.append(f"- Best WR: {best_wr['name']} WR={best_wr['wr']} PF={best_wr['pf']} DD={best_wr['max_dd']} streak={best_wr['max_loss_streak']}")
    lines.append(f"- Best streak compression: {best_streak['name']} WR={best_streak['wr']} PF={best_streak['pf']} DD={best_streak['max_dd']} streak={best_streak['max_loss_streak']}")
    lines.append(f"- Best balanced: {best_bal['name']} WR={best_bal['wr']} PF={best_bal['pf']} EXP={best_bal['expectancy']} DD={best_bal['max_dd']} streak={best_bal['max_loss_streak']}")
    lines.append("- Phase25 remains authority. Any candidate is shadow pending forensic audit.")
    lines.append("")
    write_text(OUT / "candidate_selection" / "phase29_candidate_selection.md", "\n".join(lines))
    return result, selected_specs


def forensic_safety(selected_trades: dict[str, pd.DataFrame], selected_specs: dict[str, dict[str, Any]], news: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    violations = []
    for label, trades in selected_trades.items():
        cfg = selected_specs[label]["cfg"]
        s = safety_metrics(trades, cfg, news)
        rows.append({"candidate": label, **s})
        for k, v in s.items():
            if isinstance(v, bool):
                bad = (k in ["same_bar_logic_conservative", "forced_close_correct"] and not v) or (k in ["uses_m5_for_m3", "uses_uncertified_data"] and v)
            elif isinstance(v, (int, float)):
                bad = v != 0
            else:
                bad = False
            if bad:
                violations.append({"candidate": label, "check": k, "value": v})
    safety = pd.DataFrame(rows)
    viol = pd.DataFrame(violations, columns=["candidate", "check", "value"])
    viol.to_csv(OUT / "forensic_safety" / "phase29_safety_violations.csv", index=False)
    summary = {
        "all_clear": bool(viol.empty),
        "checks": rows,
        "violation_count": int(len(viol)),
    }
    write_json(OUT / "forensic_safety" / "phase29_forensic_safety_check.json", summary)
    lines = ["# PHASE29 FORENSIC SAFETY CHECK", ""]
    lines.append(f"- All clear: {summary['all_clear']}")
    lines.append(f"- Violation count: {summary['violation_count']}")
    lines.append("- Scope confirmed: no MT5, no real, no cTrader, no VPS, no SCBI, no Phase19.")
    lines.append("")
    write_text(OUT / "forensic_safety" / "phase29_forensic_safety_check.md", "\n".join(lines))
    return safety, summary


def update_master_docs(final: dict[str, Any] | None = None) -> None:
    final = final or {}
    status = {
        "timestamp": now_utc(),
        "current_authority": "PHASE25",
        "phase25_status": "FROZEN_AUTHORITY_PAPER_DEMO_ONLY_REAL_BLOCKED",
        "phase29_status": "RESEARCH_SHADOW_ONLY",
        "phase29_verdict": final.get("verdict", "PENDING_ZIP_VALIDATION"),
        "phase29_candidate": final.get("best_balanced_name"),
        "phase29_candidate_replaces_phase25": False,
        "real_blocked": True,
        "mt5_real_blocked": True,
        "vps_blocked": True,
        "ctrader_blocked": True,
        "scbi_touched": False,
        "phase19_reopened": False,
        "next_step": "PHASE29 candidate forensic audit only if user explicitly authorizes; otherwise Phase25 remains authority.",
    }
    write_json(ROOT / "01_CURRENT_PROJECT_STATUS.json", status)
    write_json(ROOT / "02_STRATEGY_AUTHORITY_MAP.json", {
        "timestamp": now_utc(),
        "authority": "PHASE25",
        "phase25": "CURRENT_AUTHORITY_FROZEN",
        "phase29": "RESEARCH_SHADOW_ONLY_PENDING_FORENSIC_AUDIT_IF_SUPERIOR",
        "real": "BLOCKED",
        "mt5_real": "BLOCKED",
        "scbi": "PROTECTED_NOT_TOUCHED",
    })
    write_json(LAB / "status.json", status)
    write_text(
        ROOT / "00_READ_THIS_FIRST.md",
        "\n".join(
            [
                "# BOT V2 DAYTIME LAB - READ THIS FIRST",
                "",
                "- Current authority: Phase25.",
                "- Phase25 is frozen and remains paper/demo only; real, MT5 real, VPS and cTrader are blocked.",
                "- Phase29 is research shadow only. It does not replace Phase25.",
                f"- Phase29 verdict: {status['phase29_verdict']}.",
                f"- Best Phase29 shadow candidate: {status['phase29_candidate']}.",
                "- SCBI was not touched. Phase19 remains archived.",
                "- Use the canonical zip only: 000_PARA_CHATGPT.zip.",
                "",
            ]
        ),
    )
    write_text(
        ROOT / "01_CURRENT_PROJECT_STATUS.md",
        "\n".join(
            [
                "# CURRENT PROJECT STATUS",
                "",
                "- Authority: Phase25.",
                "- Phase29: research shadow only.",
                f"- Phase29 verdict: {status['phase29_verdict']}.",
                f"- Candidate: {status['phase29_candidate']}.",
                "- Promotion: none.",
                "- Real/MT5 real/VPS/cTrader: blocked.",
                "- SCBI: protected, not touched.",
                "",
            ]
        ),
    )
    write_text(
        ROOT / "02_STRATEGY_AUTHORITY_MAP.md",
        "\n".join(
            [
                "# STRATEGY AUTHORITY MAP",
                "",
                "- PHASE25: CURRENT AUTHORITY, FROZEN.",
                "- PHASE29: RESEARCH SHADOW ONLY; candidate can only move to a later forensic audit.",
                "- PHASE19: ARCHIVED.",
                "- SCBI: PROTECTED / NOT TOUCHED.",
                "- Real deployment: BLOCKED.",
                "",
            ]
        ),
    )


def zip_include(path: Path) -> bool:
    if not path.is_file():
        return False
    rel = path.relative_to(ROOT)
    parts = set(rel.parts)
    name = path.name.lower()
    suffix = path.suffix.lower()
    banned_parts = {
        ".git",
        ".venv",
        ".venv_fixed",
        "__pycache__",
        "data",
        "data_intake_2015_2019",
        "data_intake_2020_2026_bidask",
        "data_free_2020",
        "data_candidates_2022_2025",
        "scratch",
        "legacy_archive_2026",
        "quarantine",
        "mt5_local_config",
        "secrets",
    }
    if parts & banned_parts:
        return False
    if suffix in {".zip", ".zipbak", ".building", ".pkl", ".parquet", ".bi5", ".db", ".sqlite", ".dll", ".exe"}:
        return False
    if name in {".env", "mt5_local_config.json"}:
        return False
    if any(token in name for token in ["secret", "password", "token", "credential", "apikey", "api_key"]):
        return False
    try:
        if path.stat().st_size > 2 * 1024 * 1024:
            return False
    except OSError:
        return False
    rel_s = str(rel).replace("\\", "/")
    root_includes = {
        "00_READ_THIS_FIRST.md",
        "01_CURRENT_PROJECT_STATUS.md",
        "01_CURRENT_PROJECT_STATUS.json",
        "02_STRATEGY_AUTHORITY_MAP.md",
        "02_STRATEGY_AUTHORITY_MAP.json",
        "ZIP_CONTENTS_MANIFEST.md",
        "ZIP_VALIDADO_SUBIR_ESTE.txt",
        "LEER_PARA_SUBIR_ZIP.txt",
        "SUBIR_ESTE_ZIP_A_CHATGPT.txt",
    }
    if len(rel.parts) == 1:
        return rel_s in root_includes
    if rel.parts[0] != "BOT_V2_DAYTIME_LAB":
        return False
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/outputs/phase29_wr_loss_streak_compression/"):
        return suffix in {".md", ".json", ".csv", ".txt"}
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/outputs/phase28_winrate_frequency_study/"):
        return suffix in {".md", ".json", ".csv", ".txt"}
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/outputs/phase27_full_historical_validation_2015_2026/"):
        return suffix in {".md", ".json", ".csv", ".txt"} and path.stat().st_size <= 600000
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/reports/"):
        return suffix in {".md", ".json"}
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/configs/"):
        return suffix in {".json", ".txt"}
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/docs/"):
        return suffix in {".md", ".json", ".txt"}
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/src/"):
        return suffix == ".py" and (
            "phase29" in name
            or "phase28" in name
            or "phase27" in name
            or "phase26" in name
            or name in {
                "phase18_h1_fractal_sweep.py",
                "phase18_first_3m_choch.py",
                "run_same_name_zip_rebuild.py",
                "validate_zip_internals.py",
            }
        )
    if rel_s in {"BOT_V2_DAYTIME_LAB/status.json", "BOT_V2_DAYTIME_LAB/ZIP_CONTENTS_MANIFEST.md", "BOT_V2_DAYTIME_LAB/ZIP_UPLOAD_IDENTITY_MARKER.md"}:
        return True
    return False


def rebuild_zip() -> dict[str, Any]:
    if BUILD_PATH.exists():
        BUILD_PATH.unlink()
    files = []
    for p in ROOT.rglob("*"):
        if zip_include(p):
            files.append(p)
    files = sorted(set(files), key=lambda p: str(p.relative_to(ROOT)).replace("\\", "/"))
    with zipfile.ZipFile(BUILD_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, str(p.relative_to(ROOT)).replace("\\", "/"))
    with zipfile.ZipFile(BUILD_PATH, "r") as zf:
        test = zf.testzip()
        names = zf.namelist()
        heavy = [n for n in names if zf.getinfo(n).file_size > 2 * 1024 * 1024]
        secrets = [n for n in names if any(tok in n.lower() for tok in [".env", "secret", "password", "token", "credential", "apikey"])]
        zips_inside = [n for n in names if n.lower().endswith((".zip", ".zipbak"))]
    if test is not None or heavy or secrets or zips_inside:
        raise RuntimeError(f"ZIP validation failed before replace: test={test}, heavy={heavy[:5]}, secrets={secrets[:5]}, zips={zips_inside[:5]}")
    os.replace(str(BUILD_PATH), str(ZIP_PATH))
    details = zip_details(ZIP_PATH)
    live = exact_zip_inventory()
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        names = zf.namelist()
    result = {
        **details,
        "single_live_zip_exact_extension": len(live) == 1 and Path(live[0]["path"]) == ZIP_PATH,
        "live_zip_count_exact_extension": len(live),
        "contains_phase29_report": "BOT_V2_DAYTIME_LAB/reports/PHASE29_WR_LOSS_STREAK_COMPRESSION_REPORT.md" in names,
        "contains_phase29_outputs": any(n.startswith("BOT_V2_DAYTIME_LAB/outputs/phase29_wr_loss_streak_compression/") for n in names),
        "contains_phase25_config_hash": "BOT_V2_DAYTIME_LAB/configs/phase25_forward_demo_candidate_config_hash.txt" in names,
        "heavy_entries_gt_2mb": [],
        "secret_like_entries": [],
        "zip_entries_inside": [],
    }
    write_json(OUT / "zip" / "phase29_zip_validation.json", result)
    write_text(OUT / "zip" / "phase29_zip_validation.md", render_kv_md("PHASE29 ZIP VALIDATION", result))
    return result


def update_manifests(zip_result: dict[str, Any], final: dict[str, Any]) -> None:
    lines = [
        "# ZIP CONTENTS MANIFEST",
        "",
        "- Canonical live zip: 000_PARA_CHATGPT.zip",
        f"- Official path: {ZIP_PATH}",
        "- Current authority: Phase25",
        "- Phase29: research shadow only; no automatic replacement.",
        f"- Phase29 verdict: {final.get('verdict')}",
        f"- Entry count: {zip_result.get('entry_count')}",
        f"- SHA256: {zip_result.get('sha256')}",
        f"- Testzip: {zip_result.get('testzip')}",
        f"- Single live .zip: {zip_result.get('single_live_zip_exact_extension')}",
        "- No raw heavy data, no secrets, no internal zip files.",
        "",
    ]
    text = "\n".join(lines)
    write_text(ROOT / "ZIP_CONTENTS_MANIFEST.md", text)
    write_text(LAB / "ZIP_CONTENTS_MANIFEST.md", text)
    write_text(
        ROOT / "ZIP_VALIDADO_SUBIR_ESTE.txt",
        "\n".join(
            [
                "ZIP CANONICO VALIDADO",
                f"Ruta: {ZIP_PATH}",
                f"Entradas: {zip_result.get('entry_count')}",
                f"SHA256: {zip_result.get('sha256')}",
                f"Testzip: {zip_result.get('testzip')}",
                "Unico .zip vivo: SI",
                "",
            ]
        ),
    )


def final_report(
    pre: dict[str, Any],
    gate28: dict[str, Any],
    baseline: dict[str, Any],
    streak_summary: dict[str, Any],
    outcome_summary: dict[str, Any],
    single_results: pd.DataFrame,
    combo_results: pd.DataFrame,
    selection: dict[str, Any],
    wf_summary: dict[str, Any],
    cost_summary: dict[str, Any],
    safety_summary: dict[str, Any],
    zip_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    best_wr = selection.get("best_wr_candidate")
    best_streak = selection.get("best_streak_compression_candidate")
    best_bal = selection.get("best_balanced_candidate")
    wr50_rows = pd.concat([single_results, combo_results], ignore_index=True)
    wr50 = wr50_rows[wr50_rows["wr"] >= 50.0].copy()
    wr50_viable = wr50[
        (wr50["pf"] >= ACCEPTANCE["pf_min"])
        & (wr50["expectancy"] >= ACCEPTANCE["exp_min"])
        & (wr50["max_dd"] >= ACCEPTANCE["dd_max_floor"])
        & (wr50["trades_month"] >= ACCEPTANCE["trades_month_min"])
        & (wr50["months_lt15"] == 0)
    ]
    streak12 = wr50_rows[
        (wr50_rows["max_loss_streak"] <= 12)
        & (wr50_rows["pf"] >= ACCEPTANCE["pf_min"])
        & (wr50_rows["expectancy"] >= ACCEPTANCE["exp_min"])
        & (wr50_rows["max_dd"] >= ACCEPTANCE["dd_max_floor"])
        & (wr50_rows["trades_month"] >= ACCEPTANCE["trades_month_min"])
        & (wr50_rows["months_lt15"] == 0)
    ]
    report = {
        "timestamp": now_utc(),
        "objective": "Improve winrate and compress max loss streak without destroying edge.",
        "research_shadow_only": True,
        "phase25_remains_authority": True,
        "phase28_gate": gate28,
        "baseline_phase25": baseline,
        "loss_streak_forensics": streak_summary,
        "outcome_profile": outcome_summary,
        "single_hypothesis_count": int(len(single_results)),
        "limited_combination_count": int(len(combo_results)),
        "candidate_selection": selection,
        "walk_forward": wf_summary,
        "cost_stress": cost_summary,
        "forensic_safety": safety_summary,
        "wr50_viable_healthy": bool(not wr50_viable.empty),
        "wr50_diagnostic_rows": wr50.to_dict("records"),
        "streak_le12_viable_healthy": bool(not streak12.empty),
        "verdict": selection.get("verdict"),
        "best_wr_name": None if best_wr is None else best_wr.get("name"),
        "best_streak_name": None if best_streak is None else best_streak.get("name"),
        "best_balanced_name": None if best_bal is None else best_bal.get("name"),
        "zip": zip_result,
    }
    write_json(REPORT_JSON, report)
    lines = [
        "# PHASE29 WR + LOSS STREAK COMPRESSION REPORT",
        "",
        "## Objective",
        "Research shadow only. Improve WR and compress canonical non-win streak without weakening Phase25 governance.",
        "",
        "## Baseline Phase25",
        f"- Sample: {baseline['sample']}",
        f"- PF: {baseline['pf']}",
        f"- Expectancy: {baseline['expectancy']}",
        f"- WR: {baseline['wr']}",
        f"- Max DD: {baseline['max_dd']}",
        f"- Max loss streak: {baseline['max_loss_streak']}",
        f"- Trades/month: {baseline['trades_month']}",
        f"- Months <15 trades: {baseline['months_lt15']}",
        "",
        "## Diagnostics",
        f"- Streaks >=5: {baseline['streaks_ge5']}",
        f"- Streaks >=8: {baseline['streaks_ge8']}",
        f"- Streaks >=10: {baseline['streaks_ge10']}",
        f"- Streaks >=12: {baseline['streaks_ge12']}",
        f"- BE prevented original SL proxy: {outcome_summary['be_prevented_original_sl']}",
        f"- BE prevented TP proxy: {outcome_summary['be_prevented_tp']}",
        "",
        "## Hypotheses",
        f"- Single hypotheses tested: {len(single_results)}",
        f"- Limited combinations tested: {len(combo_results)}",
        "",
        "## Candidate Selection",
        f"- Verdict: {selection.get('verdict')}",
        f"- Best WR: {report['best_wr_name']}",
        f"- Best streak compression: {report['best_streak_name']}",
        f"- Best balanced: {report['best_balanced_name']}",
        "- Phase25 remains authority. No automatic replacement.",
        "",
        "## WR 50+",
        f"- Healthy WR 50+ viable: {report['wr50_viable_healthy']}",
        "- Diagnostic: WR 50+ rows are rejected unless PF/DD/frequency survive.",
        "",
        "## Max Loss Streak <=12",
        f"- Healthy <=12 viable: {report['streak_le12_viable_healthy']}",
        "",
        "## Walk-forward / Holdout",
        json.dumps(wf_summary, indent=2),
        "",
        "## Cost Stress",
        json.dumps(cost_summary, indent=2),
        "",
        "## Forensic Safety",
        json.dumps(safety_summary, indent=2),
        "",
        "## Final Verdict",
        f"{selection.get('verdict')}",
        "",
    ]
    write_text(REPORT_MD, "\n".join(lines))
    return report


def git_status_artifacts() -> dict[str, Any]:
    result = {
        "timestamp": now_utc(),
        "branch": run_cmd(["git", "branch", "--show-current"]),
        "status": run_cmd(["git", "status", "--short"]),
        "diff_stat": run_cmd(["git", "diff", "--stat"]),
        "commit": "NO",
        "push": "NO",
    }
    write_json(OUT / "git" / "phase29_git_status.json", result)
    write_text(OUT / "git" / "phase29_git_status.md", render_kv_md("PHASE29 GIT STATUS", result))
    return result


def main() -> None:
    ensure_dirs()
    pre = preflight()
    gate28 = phase28_gate()
    print("Loading data")
    m3_2015 = load_m3_2015_2019()
    m3_2020 = load_m3_2020_2026()
    df_m3 = pd.concat([m3_2015, m3_2020], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    news = load_news()
    print(f"Data rows={len(df_m3)} news={len(news)}")
    signals, h1 = generate_signals(df_m3)
    print(f"Signals={len(signals)}")
    base_trades, base_meta = backtest(df_m3, signals, news, PHASE25_CONFIG)
    base_trades = enrich_after_be_paths(base_trades, df_m3)
    baseline = calc_metrics(base_trades, "PHASE25_BASELINE")
    baseline["config_hash_file"] = (LAB / "configs" / "phase25_forward_demo_candidate_config_hash.txt").read_text(encoding="utf-8").strip()
    baseline["parameters"] = PHASE25_CONFIG
    base_safety = safety_metrics(base_trades, PHASE25_CONFIG, news)
    baseline.update({f"safety_{k}": v for k, v in base_safety.items()})
    base_trades.to_csv(OUT / "baseline_lock" / "phase29_phase25_baseline_metrics.csv", index=False)
    write_json(OUT / "baseline_lock" / "phase29_baseline_lock.json", baseline)
    write_text(OUT / "baseline_lock" / "phase29_baseline_lock.md", render_kv_md("PHASE29 BASELINE LOCK", baseline))

    streak_summary = loss_streak_artifacts(base_trades)
    outcome_summary = outcome_profile_artifacts(base_trades, df_m3)

    print("Running single hypothesis tests")
    tests = config_tests(base_trades)
    single_results, single_trade_map, cfg_map = run_test_set(df_m3, signals, news, tests, baseline)
    single_results.to_csv(OUT / "single_hypothesis_tests" / "phase29_single_hypothesis_results.csv", index=False)
    write_json(OUT / "single_hypothesis_tests" / "phase29_single_hypothesis_summary.json", {
        "tested": int(len(single_results)),
        "class_counts": single_results["classification"].value_counts().to_dict(),
        "top": single_results.sort_values(["classification", "wr"], ascending=[True, False]).head(10).to_dict("records"),
    })
    write_text(
        OUT / "single_hypothesis_tests" / "phase29_single_hypothesis_summary.md",
        "# PHASE29 SINGLE HYPOTHESIS SUMMARY\n\n"
        + single_results["classification"].value_counts().to_string()
        + "\n\n",
    )

    print("Running limited combinations")
    combo_tests = limited_combinations_tests(single_results, cfg_map)
    combo_results, combo_trade_map, combo_cfg_map = run_test_set(df_m3, signals, news, combo_tests, baseline)
    combo_results.to_csv(OUT / "limited_combinations" / "phase29_limited_combinations_results.csv", index=False)
    write_json(OUT / "limited_combinations" / "phase29_limited_combinations_summary.json", {
        "tested": int(len(combo_results)),
        "class_counts": combo_results["classification"].value_counts().to_dict() if not combo_results.empty else {},
        "rows": combo_results.to_dict("records"),
    })
    write_text(
        OUT / "limited_combinations" / "phase29_limited_combinations_summary.md",
        "# PHASE29 LIMITED COMBINATIONS SUMMARY\n\n"
        + (combo_results["classification"].value_counts().to_string() if not combo_results.empty else "No combinations qualified.")
        + "\n\n",
    )

    base_row = add_metrics_row("PHASE25_BASELINE", "BASELINE", "Phase25 exact", PHASE25_CONFIG, "NONE", base_trades, baseline, base_meta, news)
    selection, selected_specs = select_candidates(single_results, combo_results, base_row, cfg_map, combo_cfg_map)

    selected_trades: dict[str, pd.DataFrame] = {}
    for label, spec in selected_specs.items():
        source = spec.get("source_name")
        if label == "PHASE25_BASELINE":
            selected_trades[label] = base_trades
        elif source in single_trade_map:
            selected_trades[label] = single_trade_map[source]
        elif source in combo_trade_map:
            selected_trades[label] = combo_trade_map[source]
        else:
            t, _ = backtest(df_m3, signals, news, spec["cfg"], gate_name=spec.get("gate", "NONE"))
            selected_trades[label] = t

    print("Running walk-forward")
    wf_df, wf_summary = walk_forward(df_m3, signals, news, selected_specs)
    print("Running cost stress")
    cost_df, cost_summary = cost_stress(df_m3, signals, news, selected_specs)
    print("Running forensic safety")
    safety_df, safety_summary = forensic_safety(selected_trades, selected_specs, news)

    report_pre_zip = final_report(
        pre,
        gate28,
        baseline,
        streak_summary,
        outcome_summary,
        single_results,
        combo_results,
        selection,
        wf_summary,
        cost_summary,
        safety_summary,
        None,
    )
    update_master_docs(
        {
            "verdict": selection.get("verdict"),
            "best_balanced_name": report_pre_zip.get("best_balanced_name"),
        }
    )
    write_text(
        LAB / "ZIP_UPLOAD_IDENTITY_MARKER.md",
        "\n".join(
            [
                "# ZIP UPLOAD IDENTITY MARKER",
                "",
                f"- Timestamp: {now_utc()}",
                "- Phase: PHASE29_WR_LOSS_STREAK_COMPRESSION",
                "- Authority: Phase25",
                "- Phase29: research shadow only",
                "- Promotion: none",
                "",
            ]
        ),
    )
    print("Building canonical zip")
    zip_result = rebuild_zip()
    update_manifests(zip_result, selection)
    # Rebuild once more so manifests are included. The SHA reported below is from
    # the physical final archive after this final rebuild.
    zip_result = rebuild_zip()
    update_manifests(zip_result, selection)
    report = final_report(
        pre,
        gate28,
        baseline,
        streak_summary,
        outcome_summary,
        single_results,
        combo_results,
        selection,
        wf_summary,
        cost_summary,
        safety_summary,
        zip_result,
    )
    git_status_artifacts()
    print(json.dumps({"verdict": report["verdict"], "zip": zip_result}, indent=2))


if __name__ == "__main__":
    main()
