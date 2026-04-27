from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from .config import CandidateConfig, PIP_SIZE

LEVEL_GROUP_LOOKUP = {
    "pdh": "pd",
    "pdl": "pd",
    "asia_h": "asia",
    "asia_l": "asia",
    "london_h": "london",
    "london_l": "london",
}


class NewsIndex:
    def __init__(self, events: pd.DataFrame):
        frame = events.copy()
        self.coverage_start_date = pd.to_datetime(frame.attrs.get("coverage_start_date")).date() if frame.attrs.get("coverage_start_date") else None
        self.coverage_end_date = pd.to_datetime(frame.attrs.get("coverage_end_date")).date() if frame.attrs.get("coverage_end_date") else None
        if frame.empty:
            self.event_ns = np.array([], dtype=np.int64)
            self.event_names: list[str] = []
            self.events_by_date: dict[object, pd.DataFrame] = {}
            return

        timestamps = pd.to_datetime(frame["timestamp_ny"], utc=True)
        self.event_ns = timestamps.astype("int64").to_numpy()
        self.event_names = frame["event_name_normalized"].astype(str).tolist()
        frame["timestamp_ny"] = timestamps.dt.tz_convert("US/Eastern")
        frame["event_date"] = frame["timestamp_ny"].dt.date
        self.events_by_date = {
            event_date: part.reset_index(drop=True)
            for event_date, part in frame.groupby("event_date", sort=True)
        }

    @property
    def empty(self) -> bool:
        return len(self.event_ns) == 0

    def _coverage_block(self, timestamp: pd.Timestamp) -> tuple[bool, str]:
        session_date = timestamp.date()
        if self.coverage_start_date is None or self.coverage_end_date is None:
            return True, "Calendar missing"
        if session_date < self.coverage_start_date or session_date > self.coverage_end_date:
            return True, "Calendar missing"
        return False, ""

    def _same_day_events(self, timestamp: pd.Timestamp) -> pd.DataFrame:
        return self.events_by_date.get(timestamp.date(), pd.DataFrame())

    def check_sweep(self, timestamp: pd.Timestamp, mode: str) -> tuple[bool, str, str, str]:
        if mode == "none":
            return False, "", "", ""

        coverage_blocked, coverage_reason = self._coverage_block(timestamp)
        if coverage_blocked:
            return True, coverage_reason, "", "calendar_missing"

        window_minutes = {
            "sweep_plus_minus_15m": 15,
            "sweep_plus_minus_30m": 30,
            "sweep_plus_minus_60m": 60,
        }[mode]
        day_events = self._same_day_events(timestamp)
        if day_events.empty:
            return False, "", "", ""

        best_match: tuple[float, pd.Series] | None = None
        for _, row in day_events.iterrows():
            event_time = pd.Timestamp(row["timestamp_ny"]).tz_convert("US/Eastern")
            diff_minutes = abs((timestamp - event_time).total_seconds()) / 60.0
            if diff_minutes <= window_minutes:
                if best_match is None or diff_minutes < best_match[0]:
                    best_match = (diff_minutes, row)

        if best_match is None:
            return False, "", "", ""

        matched = best_match[1]
        return (
            True,
            str(matched.get("event", "") or matched.get("event_name_normalized", "News Event")),
            pd.Timestamp(matched["timestamp_ny"]).isoformat(),
            f"simplified_sweep_block_{window_minutes}m_same_day",
        )

    def context(self, timestamp: pd.Timestamp) -> dict[str, float | str | None]:
        if self.empty:
            return {
                "nearest_news_delta_minutes": None,
                "minutes_since_previous_news": None,
                "minutes_until_next_news": None,
                "nearest_news_event": "",
            }

        ts_utc = timestamp.tz_convert("UTC")
        pos = int(np.searchsorted(self.event_ns, ts_utc.value, side="left"))
        previous_delta = None
        next_delta = None
        nearest_delta = None
        nearest_event = ""

        if pos > 0:
            previous_ns = self.event_ns[pos - 1]
            previous_delta = round((ts_utc.value - previous_ns) / 60_000_000_000, 2)
        if pos < len(self.event_ns):
            next_ns = self.event_ns[pos]
            next_delta = round((next_ns - ts_utc.value) / 60_000_000_000, 2)

        candidates: list[tuple[float, int]] = []
        if previous_delta is not None:
            candidates.append((abs(previous_delta), pos - 1))
        if next_delta is not None:
            candidates.append((abs(next_delta), pos))
        if candidates:
            _, idx = min(candidates, key=lambda item: item[0])
            nearest_delta = round(abs((self.event_ns[idx] - ts_utc.value) / 60_000_000_000), 2)
            nearest_event = self.event_names[idx]

        return {
            "nearest_news_delta_minutes": nearest_delta,
            "minutes_since_previous_news": previous_delta,
            "minutes_until_next_news": next_delta,
            "nearest_news_event": nearest_event,
        }


def compute_session_levels(h1: pd.DataFrame) -> dict[object, dict[str, float]]:
    frame = h1.copy()
    frame["date"] = frame.index.date
    frame["hour"] = frame.index.hour
    # Usamos pd.unique para asegurar estabilidad en el array de fechas
    dates = sorted(pd.unique(frame["date"]))
    levels: dict[object, dict[str, float]] = {}

    for idx, current_date in enumerate(dates):
        if idx == 0:
            continue
        previous_date = dates[idx - 1]
        prev_bars = frame.loc[frame["date"] == previous_date]
        curr_bars = frame.loc[frame["date"] == current_date]
        if prev_bars.empty or curr_bars.empty:
            continue

        # --- REMEDIATION_PLAN_MINIMAL: Sunday Gap Fix ---
        is_monday = pd.Timestamp(current_date).dayofweek == 0
        is_prev_sunday = pd.Timestamp(previous_date).dayofweek == 6
        
        if is_monday and is_prev_sunday:
            # Si hoy es Lunes y ayer fue Domingo, colapsamos Viernes + Domingo para PDH/PDL
            lookback = 2
            friday_bars = pd.DataFrame()
            while idx - lookback >= 0:
                d = dates[idx - lookback]
                if pd.Timestamp(d).dayofweek < 5: # Lunes (0) a Viernes (4)
                    friday_bars = frame.loc[frame["date"] == d]
                    break
                lookback += 1
            
            if not friday_bars.empty:
                pdh = float(max(friday_bars["high"].max(), prev_bars["high"].max()))
                pdl = float(min(friday_bars["low"].min(), prev_bars["low"].min()))
            else:
                pdh = float(prev_bars["high"].max())
                pdl = float(prev_bars["low"].min())
        else:
            # Comportamiento normal (incluye Domingo usando el Viernes como previo)
            pdh = float(prev_bars["high"].max())
            pdl = float(prev_bars["low"].min())
        # ------------------------------------------------

        asia_prev = prev_bars.loc[prev_bars["hour"] >= 18]
        asia_curr = curr_bars.loc[(curr_bars["hour"] >= 18) | (curr_bars["hour"] < 2)]
        asia_all = pd.concat([asia_prev, asia_curr])
        london = curr_bars.loc[(curr_bars["hour"] >= 2) & (curr_bars["hour"] < 8)]

        levels[current_date] = {
            "pdh": pdh,
            "pdl": pdl,
            "asia_h": float(asia_all["high"].max()) if not asia_all.empty else pdh,
            "asia_l": float(asia_all["low"].min()) if not asia_all.empty else pdl,
            "london_h": float(london["high"].max()) if not london.empty else pdh,
            "london_l": float(london["low"].min()) if not london.empty else pdl,
        }
    return levels


def detect_sweeps(h1: pd.DataFrame, *, start_date: str, end_date: str) -> pd.DataFrame:
    start_day = pd.Timestamp(start_date).date()
    end_day = pd.Timestamp(end_date).date()
    levels = compute_session_levels(h1)
    frame = h1.copy()
    frame["date"] = frame.index.date
    rows: list[dict[str, object]] = []

    for timestamp, bar in frame.iterrows():
        session_date = bar["date"]
        if session_date < start_day or session_date > end_day or session_date not in levels:
            continue

        level_map = levels[session_date]
        open_price = float(bar["open"])
        high_price = float(bar["high"])
        low_price = float(bar["low"])
        close_price = float(bar["close"])

        for level_name in ("pdl", "asia_l", "london_l"):
            level_price = float(level_map[level_name])
            if low_price < level_price and close_price > level_price:
                rows.append(
                    {
                        "session_date": str(session_date),
                        "sweep_time": timestamp,
                        "direction": "long",
                        "level_name": level_name,
                        "level_group": LEVEL_GROUP_LOOKUP[level_name],
                        "level_price": level_price,
                        "sweep_extreme": low_price,
                        "h1_open": open_price,
                        "h1_high": high_price,
                        "h1_low": low_price,
                        "h1_close": close_price,
                    }
                )

        for level_name in ("pdh", "asia_h", "london_h"):
            level_price = float(level_map[level_name])
            if high_price > level_price and close_price < level_price:
                rows.append(
                    {
                        "session_date": str(session_date),
                        "sweep_time": timestamp,
                        "direction": "short",
                        "level_name": level_name,
                        "level_group": LEVEL_GROUP_LOOKUP[level_name],
                        "level_price": level_price,
                        "sweep_extreme": high_price,
                        "h1_open": open_price,
                        "h1_high": high_price,
                        "h1_low": low_price,
                        "h1_close": close_price,
                    }
                )

    sweeps = pd.DataFrame(rows)
    if sweeps.empty:
        return sweeps
    sweeps = sweeps.sort_values("sweep_time").reset_index(drop=True)
    sweeps["sweep_id"] = [
        f"SWEEP_{idx:06d}_{ts.strftime('%Y%m%d_%H%M')}"
        for idx, ts in enumerate(sweeps["sweep_time"], start=1)
    ]
    return sweeps


def _body_strength(row: pd.Series) -> float:
    high_price = float(row["high"])
    low_price = float(row["low"])
    price_range = high_price - low_price
    if price_range <= 0:
        return 0.0
    return abs(float(row["close"]) - float(row["open"])) / price_range


def find_confirmation_candidate(m5: pd.DataFrame, sweep: pd.Series, config: CandidateConfig) -> dict[str, object]:
    sweep_time = pd.Timestamp(sweep["sweep_time"])
    search_start = sweep_time + pd.Timedelta(hours=config.confirmation_window_start_hours)
    search_end = sweep_time + pd.Timedelta(hours=config.confirmation_window_end_hours)
    window = m5.loc[(m5.index >= search_start) & (m5.index <= search_end)]
    if window.empty:
        return {"status": "no_scbi_window"}

    direction = str(sweep["direction"])
    level_price = float(sweep["level_price"])
    qualifying: list[tuple[int, float, float]] = []
    for position in range(len(window)):
        bar = window.iloc[position]
        close_price = float(bar["close"])
        trigger = close_price > level_price if direction == "long" else close_price < level_price
        if not trigger:
            continue

        body_strength = _body_strength(bar)
        if config.confirmation_mode == "close_reclaim_body_strength":
            supportive_body = close_price > float(bar["open"]) if direction == "long" else close_price < float(bar["open"])
            if not supportive_body or body_strength < config.body_strength_threshold:
                continue

        reclaim_distance = close_price - level_price if direction == "long" else level_price - close_price
        qualifying.append((position, reclaim_distance, body_strength))

    if not qualifying:
        return {"status": "no_scbi_found"}

    eligible = [item for item in qualifying if item[0] + 1 < len(window)]
    if not eligible:
        return {"status": "no_entry_bar_after_scbi"}

    if config.confirmation_pick == "first":
        chosen = eligible[0]
    else:
        chosen = max(eligible, key=lambda item: (item[1], item[2], -item[0]))

    signal_pos = int(chosen[0])
    entry_pos = signal_pos + 1
    signal_row = window.iloc[signal_pos]
    entry_row = window.iloc[entry_pos]
    return {
        "status": "tradable",
        "signal_time": window.index[signal_pos],
        "entry_time": window.index[entry_pos],
        "entry_open": float(entry_row["open"]),
        "signal_body_strength": round(chosen[2], 4),
        "reclaim_distance_pips": round(chosen[1] / PIP_SIZE, 4),
        "signal_open": float(signal_row["open"]),
        "signal_close": float(signal_row["close"]),
    }


def _pnl_r(direction: str, entry_price: float, stop_loss: float, exit_price: float) -> float:
    risk_distance = entry_price - stop_loss if direction == "long" else stop_loss - entry_price
    if risk_distance <= 0:
        return 0.0
    if direction == "long":
        return round((exit_price - entry_price) / risk_distance, 4)
    return round((entry_price - exit_price) / risk_distance, 4)


def simulate_trade(m5: pd.DataFrame, sweep: pd.Series, candidate: dict[str, object], news_index: NewsIndex, config: CandidateConfig) -> dict[str, object]:
    direction = str(sweep["direction"])
    sweep_extreme = float(sweep["sweep_extreme"])
    entry_time = pd.Timestamp(candidate["entry_time"])

    if direction == "long":
        entry_price = float(candidate["entry_open"]) + (config.long_entry_buffer_pips * PIP_SIZE)
        stop_loss = sweep_extreme - (config.sl_buffer_pips * PIP_SIZE)
        risk_distance = entry_price - stop_loss
        take_profit = entry_price + (config.tp_r * risk_distance)
    else:
        entry_price = float(candidate["entry_open"]) + (config.short_entry_buffer_pips * PIP_SIZE)
        stop_loss = sweep_extreme + (config.sl_buffer_pips * PIP_SIZE)
        risk_distance = stop_loss - entry_price
        take_profit = entry_price - (config.tp_r * risk_distance)

    if risk_distance <= 0 or risk_distance < (config.min_risk_pips * PIP_SIZE):
        return {"status": "invalid_risk"}

    end_time = entry_time + pd.Timedelta(hours=config.timeout_hours)
    future = m5.loc[(m5.index >= entry_time) & (m5.index <= end_time)]
    exit_time = entry_time
    exit_price = entry_price
    exit_reason = "timeout"
    pnl_r = 0.0

    for timestamp, row in future.iterrows():
        high_price = float(row["high"])
        low_price = float(row["low"])
        if direction == "long":
            if low_price <= stop_loss:
                exit_time = timestamp
                exit_price = stop_loss
                exit_reason = "sl_hit"
                pnl_r = -1.0
                break
            if high_price >= take_profit:
                exit_time = timestamp
                exit_price = take_profit
                exit_reason = "tp_hit"
                pnl_r = round(config.tp_r, 4)
                break
        else:
            if high_price >= stop_loss:
                exit_time = timestamp
                exit_price = stop_loss
                exit_reason = "sl_hit"
                pnl_r = -1.0
                break
            if low_price <= take_profit:
                exit_time = timestamp
                exit_price = take_profit
                exit_reason = "tp_hit"
                pnl_r = round(config.tp_r, 4)
                break
    else:
        if not future.empty:
            exit_time = future.index[-1]
            exit_price = float(future.iloc[-1]["close"])
        pnl_r = _pnl_r(direction, entry_price, stop_loss, exit_price)

    news_context = news_index.context(pd.Timestamp(sweep["sweep_time"]))
    hold_minutes = round((exit_time - entry_time).total_seconds() / 60.0, 2)
    return {
        "status": "trade_executed",
        "session_date": str(sweep["session_date"]),
        "sweep_id": str(sweep["sweep_id"]),
        "sweep_time": pd.Timestamp(sweep["sweep_time"]).isoformat(),
        "signal_time": pd.Timestamp(candidate["signal_time"]).isoformat(),
        "entry_time": entry_time.isoformat(),
        "exit_time": exit_time.isoformat(),
        "direction": direction,
        "level_name": str(sweep["level_name"]),
        "level_group": str(sweep["level_group"]),
        "level_price": round(float(sweep["level_price"]), 5),
        "sweep_extreme": round(sweep_extreme, 5),
        "entry_price": round(entry_price, 5),
        "sl": round(stop_loss, 5),
        "tp": round(take_profit, 5),
        "risk_pips": round(risk_distance / PIP_SIZE, 4),
        "exit_price": round(exit_price, 5),
        "exit_reason": exit_reason,
        "pnl_r": round(pnl_r, 4),
        "hold_minutes": hold_minutes,
        "signal_body_strength": candidate["signal_body_strength"],
        "reclaim_distance_pips": candidate["reclaim_distance_pips"],
        "news_mode": config.news_mode,
        **news_context,
    }


def run_baseline_truth_model(config: CandidateConfig, *, h1: pd.DataFrame, m5: pd.DataFrame, news: pd.DataFrame) -> dict[str, object]:
    sweeps = detect_sweeps(h1, start_date=config.start_date, end_date=config.end_date)
    news_index = NewsIndex(news)
    stats: Counter[str] = Counter()
    audit_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    traded_days: set[str] = set()

    if sweeps.empty:
        return {
            "config": asdict(config),
            "stats": {},
            "sweeps": sweeps,
            "sweep_audit": pd.DataFrame(),
            "trades": pd.DataFrame(),
            "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        }

    allowed_groups = set(config.allowed_level_groups)
    for _, sweep in sweeps.iterrows():
        sweep_time = pd.Timestamp(sweep["sweep_time"])
        session_date = str(sweep["session_date"])
        base_audit = {
            "variant_id": config.variant_id,
            "session_date": session_date,
            "sweep_id": str(sweep["sweep_id"]),
            "sweep_time": sweep_time.isoformat(),
            "direction": str(sweep["direction"]),
            "level_name": str(sweep["level_name"]),
            "level_group": str(sweep["level_group"]),
            "status": "",
            "detail": "",
        }

        if str(sweep["level_group"]) not in allowed_groups:
            stats["level_filtered"] += 1
            audit_rows.append({**base_audit, "status": "LEVEL_FILTERED", "detail": config.level_profile})
            continue

        stats["sweeps_considered"] += 1

        blocked, block_reason, event_time, rule_used = news_index.check_sweep(sweep_time, config.news_mode)
        if blocked:
            stats["news_blocked"] += 1
            audit_rows.append({**base_audit, "status": "NEWS_BLOCKED", "detail": f"{block_reason} | {rule_used} | {event_time}"})
            continue

        if session_date in traded_days:
            stats["daily_limit_skipped"] += 1
            audit_rows.append({**base_audit, "status": "DAILY_LIMIT_SKIPPED", "detail": "1 trade per day"})
            continue

        candidate = find_confirmation_candidate(m5, sweep, config)
        candidate_status = str(candidate["status"])
        if candidate_status != "tradable":
            stats[candidate_status] += 1
            audit_rows.append({**base_audit, "status": candidate_status.upper(), "detail": ""})
            continue

        trade = simulate_trade(m5, sweep, candidate, news_index, config)
        if trade["status"] == "invalid_risk":
            stats["invalid_risk"] += 1
            audit_rows.append({**base_audit, "status": "INVALID_RISK", "detail": ""})
            continue

        traded_days.add(session_date)
        stats["trades_executed"] += 1
        trade_rows.append(trade)
        audit_rows.append({**base_audit, "status": "TRADE_EXECUTED", "detail": f"{trade['entry_time']} -> {trade['exit_reason']} ({trade['pnl_r']}R)"})

    trades = pd.DataFrame(trade_rows).sort_values("entry_time") if trade_rows else pd.DataFrame()
    audit = pd.DataFrame(audit_rows)
    return {
        "config": asdict(config),
        "stats": dict(stats),
        "sweeps": sweeps,
        "sweep_audit": audit,
        "trades": trades,
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
