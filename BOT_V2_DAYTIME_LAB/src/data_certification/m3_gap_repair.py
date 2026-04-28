from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_certification.forex_market_calendar import (
    interval_in_phase18_window,
    interval_in_phase19_window,
    interval_in_rollover,
    interval_in_user_window,
    interval_market_expected_open,
    missing_bar_times,
    to_ny,
)
from news_guard_strict import normalize_news


M3_FREQ_MINUTES = 3
NY_TZ = "America/New_York"


def load_m3_bid_ask(bid_path: str | Path, ask_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    bid = pd.read_csv(bid_path)
    ask = pd.read_csv(ask_path)
    bid["timestamp"] = pd.to_datetime(bid["timestamp"], utc=True, errors="coerce")
    ask["timestamp"] = pd.to_datetime(ask["timestamp"], utc=True, errors="coerce")
    return bid.sort_values("timestamp").reset_index(drop=True), ask.sort_values("timestamp").reset_index(drop=True)


def build_m3_gap_table(bid: pd.DataFrame, ask: pd.DataFrame) -> pd.DataFrame:
    bid_ts = pd.DatetimeIndex(pd.to_datetime(bid["timestamp"], utc=True).dropna().unique()).sort_values()
    ask_ts = pd.DatetimeIndex(pd.to_datetime(ask["timestamp"], utc=True).dropna().unique()).sort_values()
    union_ts = bid_ts.union(ask_ts).sort_values()
    rows = []
    for i in range(1, len(union_ts)):
        prev_ts = union_ts[i - 1]
        next_ts = union_ts[i]
        gap_minutes = (next_ts - prev_ts).total_seconds() / 60.0
        if gap_minutes <= M3_FREQ_MINUTES:
            continue
        first_missing = prev_ts + pd.Timedelta(minutes=M3_FREQ_MINUTES)
        last_missing = next_ts - pd.Timedelta(minutes=M3_FREQ_MINUTES)
        expected = missing_bar_times(first_missing, last_missing, M3_FREQ_MINUTES)
        bid_missing = [ts for ts in expected if ts not in bid_ts]
        ask_missing = [ts for ts in expected if ts not in ask_ts]
        rows.append(
            {
                "gap_id": len(rows),
                "prev_known_utc": prev_ts,
                "next_known_utc": next_ts,
                "start_utc": first_missing,
                "end_utc": last_missing,
                "duration_minutes": gap_minutes,
                "missing_m3_bars": int(len(expected)),
                "bid_missing": bool(bid_missing),
                "ask_missing": bool(ask_missing),
                "both_missing": bool(bid_missing and ask_missing and len(bid_missing) == len(expected) and len(ask_missing) == len(expected)),
            }
        )
    return pd.DataFrame(rows)


def _news_near_interval(news: pd.DataFrame, start_utc, end_utc, buffer_minutes: int) -> bool:
    if news is None or news.empty:
        return False
    start = pd.Timestamp(start_utc).tz_convert("UTC") - pd.Timedelta(minutes=buffer_minutes)
    end = pd.Timestamp(end_utc).tz_convert("UTC") + pd.Timedelta(minutes=buffer_minutes)
    high = news[(news["currency"].isin(["USD", "EUR"])) & (news["impact_level"] == "HIGH")]
    if high.empty:
        return False
    return bool(((high["timestamp_utc"] >= start) & (high["timestamp_utc"] <= end)).any())


def _base_missing_classification(row: pd.Series) -> str | None:
    if row["bid_missing"] and not row["ask_missing"]:
        return "MISSING_BID_ONLY"
    if row["ask_missing"] and not row["bid_missing"]:
        return "MISSING_ASK_ONLY"
    if row["both_missing"]:
        return "MISSING_BOTH"
    return None


def classify_gap(row: pd.Series, news: pd.DataFrame | None = None) -> dict:
    times = missing_bar_times(row["start_utc"], row["end_utc"], M3_FREQ_MINUTES)
    start_ny = to_ny(row["start_utc"])
    end_ny = to_ny(row["end_utc"])
    market_open = interval_market_expected_open(times)
    in_user = interval_in_user_window(times)
    in_phase19 = interval_in_phase19_window(times)
    in_phase18 = interval_in_phase18_window(times)
    in_rollover = interval_in_rollover(times)
    near_news_30 = _news_near_interval(news, row["start_utc"], row["end_utc"], 30) if news is not None else False
    near_news_60 = _news_near_interval(news, row["start_utc"], row["end_utc"], 60) if news is not None else False

    missing_class = _base_missing_classification(row)
    classification = missing_class or "MISSING_BOTH"
    severity = "CRITICAL_BLOCK_CERTIFICATION"
    action = "CERTIFICATION_BLOCKED"

    if not market_open:
        classification = "WEEKEND_MARKET_CLOSED"
        severity = "IGNORE_SAFE"
        action = "RECLASSIFIED_MARKET_CLOSED"
    elif in_rollover and not in_phase19 and not in_phase18:
        classification = "ROLLOVER_MAINTENANCE"
        severity = "WARNING_MASK_SESSION" if in_user else "IGNORE_SAFE"
        action = "RECLASSIFIED_OUTSIDE_SCOPE"
    elif not in_user:
        classification = "OUTSIDE_USER_WINDOW"
        severity = "IGNORE_SAFE"
        action = "RECLASSIFIED_OUTSIDE_SCOPE"
    elif near_news_60 and (in_phase19 or in_phase18):
        classification = "NEWS_WINDOW_GAP"
        severity = "CRITICAL_MASK_DAY"
        action = "MASKED_DAY"
    elif in_phase18:
        classification = "INTRADAY_CRITICAL_PHASE18"
        severity = "CRITICAL_MASK_DAY"
        action = "MASKED_DAY"
    elif in_phase19:
        classification = "INTRADAY_CRITICAL_PHASE19"
        severity = "CRITICAL_MASK_DAY"
        action = "MASKED_DAY"
    elif in_user:
        classification = "OUTSIDE_PHASE19_WINDOW"
        severity = "WARNING_MASK_SESSION"
        action = "MASKED_SESSION"
    else:
        classification = "UNKNOWN_CRITICAL_GAP"
        severity = "CRITICAL_BLOCK_CERTIFICATION"
        action = "UNREPAIRABLE_SOURCE_GAP"

    if missing_class in {"MISSING_BID_ONLY", "MISSING_ASK_ONLY"} and severity != "IGNORE_SAFE":
        classification = missing_class
        severity = "CRITICAL_MASK_DAY" if in_phase19 or in_phase18 else "WARNING_MASK_SESSION"
        action = "MASKED_DAY" if in_phase19 or in_phase18 else "MASKED_SESSION"

    return {
        "start_ny": start_ny.isoformat(),
        "end_ny": end_ny.isoformat(),
        "weekday_ny": start_ny.day_name(),
        "market_expected_open": market_open,
        "in_user_window_07_20": in_user,
        "in_phase19_window_08_1630": in_phase19,
        "in_phase18_window_08_11": in_phase18,
        "in_rollover_17_19": in_rollover,
        "near_high_impact_news_30m": near_news_30,
        "near_high_impact_news_60m": near_news_60,
        "classification": classification,
        "severity": severity,
        "recommended_action": action,
    }


def classify_m3_gaps(
    bid_path: str | Path,
    ask_path: str | Path,
    news_path: str | Path | None = None,
) -> pd.DataFrame:
    bid, ask = load_m3_bid_ask(bid_path, ask_path)
    gaps = build_m3_gap_table(bid, ask)
    news = None
    if news_path and Path(news_path).exists():
        news = normalize_news(pd.read_csv(news_path))
    if gaps.empty:
        return gaps
    classified = []
    for _, row in gaps.iterrows():
        info = classify_gap(row, news)
        payload = row.to_dict()
        payload.update(info)
        classified.append(payload)
    out = pd.DataFrame(classified)
    for col in ["start_utc", "end_utc", "prev_known_utc", "next_known_utc"]:
        out[col] = pd.to_datetime(out[col], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    return out


def build_repair_actions(classified: pd.DataFrame) -> pd.DataFrame:
    if classified.empty:
        return pd.DataFrame(columns=["gap_id", "action", "reason"])
    return pd.DataFrame(
        {
            "gap_id": classified["gap_id"],
            "action": classified["recommended_action"],
            "reason": classified["classification"],
        }
    )


def build_data_quality_mask(classified: pd.DataFrame, start_utc, end_utc) -> pd.DataFrame:
    start_ny = pd.Timestamp(start_utc).tz_convert(NY_TZ).date()
    end_ny = pd.Timestamp(end_utc).tz_convert(NY_TZ).date()
    dates = pd.date_range(start_ny, end_ny, freq="D")
    rows = []
    if classified.empty:
        classified = pd.DataFrame()
    df = classified.copy()
    if not df.empty:
        df["date_ny"] = df["start_ny"].astype(str).str.slice(0, 10)
    for date in dates:
        date_str = date.date().isoformat()
        day = df[df["date_ny"] == date_str] if not df.empty else pd.DataFrame()
        critical = day[day["severity"].isin(["CRITICAL_MASK_DAY", "CRITICAL_BLOCK_CERTIFICATION"])] if not day.empty else pd.DataFrame()
        warning = day[day["severity"] == "WARNING_MASK_SESSION"] if not day.empty else pd.DataFrame()
        phase18_bad = bool((critical["in_phase18_window_08_11"]).any()) if not critical.empty else False
        phase19_bad = bool((critical["in_phase19_window_08_1630"]).any()) if not critical.empty else False
        user_bad = bool((critical["in_user_window_07_20"]).any()) if not critical.empty else False
        unknown_bad = bool((critical["classification"] == "UNKNOWN_CRITICAL_GAP").any()) if not critical.empty else False
        reasons = sorted(set(critical["classification"].astype(str).tolist() + warning["classification"].astype(str).tolist())) if not day.empty else []
        if unknown_bad:
            phase18_bad = True
            phase19_bad = True
            user_bad = True
        rows.append(
            {
                "date_ny": date_str,
                "session_start_ny": f"{date_str}T07:00:00",
                "session_end_ny": f"{date_str}T20:00:00",
                "phase18_window_ok": not phase18_bad,
                "phase19_window_ok": not phase19_bad,
                "user_window_ok": not user_bad,
                "blocked_reason": ";".join(reasons),
                "critical_gap_count": int(len(critical)),
                "warning_gap_count": int(len(warning)),
                "allow_phase18": not phase18_bad,
                "allow_phase19_repaired": not phase19_bad,
                "allow_user_window": not user_bad,
            }
        )
    return pd.DataFrame(rows)


def summarize_gap_classification(classified: pd.DataFrame) -> dict:
    if classified.empty:
        return {
            "total_gaps": 0,
            "phase19_critical_gaps": 0,
            "phase18_critical_gaps": 0,
            "critical_block_certification": 0,
        }
    severity_counts = classified["severity"].value_counts().to_dict()
    classification_counts = classified["classification"].value_counts().to_dict()
    phase19_critical = classified[
        classified["in_phase19_window_08_1630"] & classified["severity"].isin(["CRITICAL_MASK_DAY", "CRITICAL_BLOCK_CERTIFICATION"])
    ]
    phase18_critical = classified[
        classified["in_phase18_window_08_11"] & classified["severity"].isin(["CRITICAL_MASK_DAY", "CRITICAL_BLOCK_CERTIFICATION"])
    ]
    return {
        "total_gaps": int(len(classified)),
        "severity_counts": {str(k): int(v) for k, v in severity_counts.items()},
        "classification_counts": {str(k): int(v) for k, v in classification_counts.items()},
        "phase19_critical_gaps": int(len(phase19_critical)),
        "phase18_critical_gaps": int(len(phase18_critical)),
        "critical_block_certification": int((classified["severity"] == "CRITICAL_BLOCK_CERTIFICATION").sum()),
        "days_phase19_blocked": int(phase19_critical["start_ny"].astype(str).str.slice(0, 10).nunique()) if not phase19_critical.empty else 0,
        "days_phase18_blocked": int(phase18_critical["start_ny"].astype(str).str.slice(0, 10).nunique()) if not phase18_critical.empty else 0,
    }
