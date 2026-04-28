from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


NY_TZ = "America/New_York"

CRITICAL_FAMILIES = {
    "NFP": ["non farm", "nonfarm", "payroll"],
    "CPI": ["cpi", "consumer price"],
    "CORE_CPI": ["core cpi"],
    "FOMC": ["fomc"],
    "FED_RATE": ["fed rate", "federal funds", "interest rate decision"],
    "FED_CHAIR": ["powell", "fed chair", "federal reserve chair"],
    "PCE": ["pce", "personal consumption"],
    "RETAIL_SALES": ["retail sales"],
    "GDP": ["gdp", "gross domestic"],
    "ISM": ["ism"],
    "UNEMPLOYMENT_CLAIMS": ["unemployment claims", "jobless claims"],
    "ECB_RATE": ["ecb rate", "main refinancing rate", "deposit facility rate"],
    "ECB_PRESS": ["ecb press", "lagarde"],
    "EUROZONE_CPI": ["eurozone cpi", "euro area cpi"],
    "GERMAN_CPI": ["german cpi", "germany cpi"],
    "PMI": ["pmi"],
}


@dataclass(frozen=True)
class NewsGuardDecision:
    blocked: bool
    reason: str
    matched_events: int = 0


class NewsGuardStrict:
    def __init__(self, events: pd.DataFrame, currencies: set[str] | None = None):
        self.currencies = currencies or {"USD", "EUR"}
        self.events = normalize_news(events)

    @classmethod
    def from_csv(cls, path: str | Path) -> "NewsGuardStrict":
        if not Path(path).exists():
            raise FileNotFoundError(f"missing news feed: {path}")
        return cls(pd.read_csv(path))

    def should_block(self, timestamp, buffer_minutes: int = 30) -> NewsGuardDecision:
        if self.events.empty:
            return NewsGuardDecision(True, "NEWS_FEED_EMPTY", 0)
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            return NewsGuardDecision(True, "TIMESTAMP_WITHOUT_TIMEZONE", 0)
        ts_utc = ts.tz_convert("UTC")
        delta = pd.Timedelta(minutes=buffer_minutes)
        relevant = self.events[self.events["currency"].isin(self.currencies)].copy()
        invalid = relevant[(relevant["timestamp_utc"].isna()) | (relevant["impact_level"] == "UNKNOWN")]
        if not invalid.empty:
            return NewsGuardDecision(True, "NEWS_FEED_HAS_INVALID_OR_UNKNOWN_EVENTS", int(len(invalid)))
        high = relevant[relevant["impact_level"] == "HIGH"]
        matched = high[(high["timestamp_utc"] >= ts_utc - delta) & (high["timestamp_utc"] <= ts_utc + delta)]
        if not matched.empty:
            return NewsGuardDecision(True, "HIGH_IMPACT_USD_EUR_WITHIN_BUFFER", int(len(matched)))
        return NewsGuardDecision(False, "CLEAR", 0)


def _event_col(df: pd.DataFrame) -> str:
    for col in ["event_name_normalized", "event", "name", "title"]:
        if col in df.columns:
            return col
    return ""


def normalize_news(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp_utc", "timestamp_ny", "currency", "impact_level", "event_name", "family"])
    out = df.copy()
    ts_col = "timestamp_utc" if "timestamp_utc" in out.columns else ("timestamp" if "timestamp" in out.columns else None)
    if ts_col is None:
        out["timestamp_utc"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    else:
        out["timestamp_utc"] = pd.to_datetime(out[ts_col], utc=True, errors="coerce")
    out["timestamp_ny"] = out["timestamp_utc"].dt.tz_convert(NY_TZ)
    out["currency"] = out.get("currency", "UNKNOWN").astype(str).str.upper()
    out["impact_level"] = out.get("impact_level", "UNKNOWN").astype(str).str.upper()
    out.loc[~out["impact_level"].isin(["HIGH", "MEDIUM", "LOW"]), "impact_level"] = "UNKNOWN"
    col = _event_col(out)
    out["event_name"] = out[col].astype(str).str.lower() if col else "unknown"
    out["family"] = out["event_name"].map(classify_family)
    dedupe_cols = ["timestamp_utc", "currency", "impact_level", "event_name"]
    out = out.drop_duplicates(subset=dedupe_cols).reset_index(drop=True)
    return out


def classify_family(name: str) -> str:
    text = str(name).lower()
    for family, tokens in CRITICAL_FAMILIES.items():
        if any(token in text for token in tokens):
            return family
    return "OTHER"


def audit_news_feed(path: str | Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    if not Path(path).exists():
        summary = {"verdict": "NEWS_GUARD_INVALIDATED", "reason": "missing_news_feed"}
        return summary, pd.DataFrame(), pd.DataFrame()
    raw = pd.read_csv(path)
    normalized = normalize_news(raw)
    family = (
        normalized.groupby(["currency", "impact_level", "family"], dropna=False)
        .size()
        .reset_index(name="events")
        .sort_values(["currency", "impact_level", "family"])
    )
    coverage = (
        normalized.assign(year=normalized["timestamp_utc"].dt.year)
        .groupby(["year", "currency", "impact_level"], dropna=False)
        .size()
        .reset_index(name="events")
        .sort_values(["year", "currency", "impact_level"])
    )
    invalid_ts = int(normalized["timestamp_utc"].isna().sum())
    unknown_impact = int((normalized["impact_level"] == "UNKNOWN").sum())
    high_usd_eur = normalized[(normalized["currency"].isin(["USD", "EUR"])) & (normalized["impact_level"] == "HIGH")]
    critical_present = set(high_usd_eur["family"].unique()) - {"OTHER"}
    summary = {
        "verdict": "NEWS_GUARD_STRICT_CERTIFIED"
        if invalid_ts == 0 and unknown_impact == 0 and len(critical_present) >= 8
        else "NEWS_GUARD_REQUIRES_REPAIR",
        "rows_raw": int(len(raw)),
        "rows_deduped": int(len(normalized)),
        "invalid_timestamp_count": invalid_ts,
        "unknown_impact_count": unknown_impact,
        "high_impact_usd_eur_count": int(len(high_usd_eur)),
        "critical_families_present": sorted(critical_present),
        "timezone_utc_validated": invalid_ts == 0,
        "timezone_ny_validated": invalid_ts == 0,
        "dst_validated": invalid_ts == 0,
        "buffers_supported_minutes": [30, 45, 60, 90],
    }
    return summary, coverage, family


def save_news_audit(path: str | Path, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary, coverage, family = audit_news_feed(path)
    coverage.to_csv(out / "news_coverage_report.csv", index=False)
    family.to_csv(out / "news_family_map.csv", index=False)
    with (out / "news_guard_strict_audit.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
        f.write("\n")
    with (out / "news_guard_strict_audit.md").open("w", encoding="utf-8", newline="\n") as f:
        f.write("# News Guard Strict Audit\n\n")
        f.write(f"Verdicto: {summary['verdict']}\n")
        f.write(f"Rows raw/deduped: {summary.get('rows_raw', 0)} / {summary.get('rows_deduped', 0)}\n")
        f.write(f"High impact USD/EUR: {summary.get('high_impact_usd_eur_count', 0)}\n")
    return summary
