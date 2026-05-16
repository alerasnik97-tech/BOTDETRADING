from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
RESULTS_DIR = ROOT / "results"

GLOBAL_HISTORY_CSV = RESULTS_DIR / "SCBI_2020_2025_DURABILITY" / "trades_baseline.csv"
CORE_HISTORY_CSV = RESULTS_DIR / "SCBI_CORE_STAGE2" / "core_stage2_trades.csv"
GLOBAL_FORWARD_CSV = RESULTS_DIR / "SCBI_FORWARD_LEDGER.csv"
CORE_FORWARD_CSV = RESULTS_DIR / "SCBI_CORE_PHASE1" / "core_phase1_ledger.csv"
SCOREBOARD_CSV = RESULTS_DIR / "SCBI_DUAL_LINE_SCOREBOARD.csv"
TRIBUNAL_JSON = RESULTS_DIR / "SCBI_FORWARD_TRIBUNAL_SUMMARY.json"
VALIDATION_JSON = RESULTS_DIR / "SCBI_SEQUENTIAL_EVIDENCE_VALIDATION.json"
STATUS_JSON = RESULTS_DIR / "SCBI_SEQUENTIAL_EVIDENCE_STATUS.json"
TRACE_CSV = RESULTS_DIR / "SCBI_SEQUENTIAL_EVIDENCE_TRACE.csv"
DAILY_CSV = RESULTS_DIR / "SCBI_SEQUENTIAL_EVIDENCE_DAILY.csv"

MAX_PREFIX_CAP = 120
MIN_REFERENCE_WINDOWS = 60
SCORING_VERSION_RECALIBRATED = "RECALIBRATED_V2"
SCORING_VERSION_REFINED = "MATERIAL_REFINED_V3"
SCORING_VERSION_LEGACY = "LEGACY_V1"
STATE_ORDER = {
    "SEQUENTIAL_MODEL_NOT_RELIABLE": 0,
    "EVIDENCE_MATERIALLY_UNFAVORABLE": 1,
    "EVIDENCE_EARLY_WARNING": 2,
    "EVIDENCE_TENSE_BUT_NOT_ALARMING": 3,
    "EVIDENCE_STILL_THIN": 4,
    "EVIDENCE_ACCUMULATING_NORMALLY": 5,
}
WARNING_STATES = {"EVIDENCE_EARLY_WARNING", "EVIDENCE_MATERIALLY_UNFAVORABLE"}
EARLY_OR_WORSE_STATES = {"EVIDENCE_EARLY_WARNING", "EVIDENCE_MATERIALLY_UNFAVORABLE"}
EPSILON = 1e-6
DELTA_FLAT_EPSILON = 1e-9

LINE_CONFIGS: dict[str, dict[str, Any]] = {
    "SCBI_M5_GLOBAL": {
        "historical_path": GLOBAL_HISTORY_CSV,
        "historical_pnl_col": "pnl_r",
        "forward_path": GLOBAL_FORWARD_CSV,
        "history_time_col": "exit_time",
        "forward_kind": "GLOBAL",
        "history_note": "Baseline historica canonica GLOBAL usando pnl_r.",
        "forward_note": "Ledger oficial GLOBAL filtrado a PAPER_EXIT con pnl_r no nulo.",
    },
    "SCBI_CORE": {
        "historical_path": CORE_HISTORY_CSV,
        "historical_pnl_col": "pnl_r_dynamic",
        "forward_path": CORE_FORWARD_CSV,
        "history_time_col": "exit_time",
        "forward_kind": "CORE",
        "history_note": "Baseline historica canonica CORE usando pnl_r_dynamic.",
        "forward_note": "Ledger oficial CORE filtrado a IDs CORE_ con pnl_r no nulo.",
    },
}


@dataclass(frozen=True)
class PrefixReference:
    n: int
    starts: np.ndarray
    expectancy: np.ndarray
    max_dd: np.ndarray
    pf: np.ndarray
    win_rate: np.ndarray

    @property
    def window_count(self) -> int:
        return int(len(self.starts))


@dataclass(frozen=True)
class ReferenceModel:
    line_name: str
    history: pd.DataFrame
    history_hash: str
    max_prefix: int
    prefix_map: dict[int, PrefixReference]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Falta fuente requerida: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def as_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def safe_round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def geometric_mean_unit(values: list[float]) -> float:
    if not values:
        return 0.0
    arr = np.clip(np.asarray(values, dtype=float), EPSILON, 1.0)
    return float(math.exp(float(np.mean(np.log(arr)))))


def clamp_unit(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def compute_pf(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    wins = values[values > 0.0]
    losses = values[values <= 0.0]
    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    if gross_loss <= 0.0:
        return 999.0
    return gross_profit / gross_loss


def compute_max_dd(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    equity = np.cumsum(values)
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity - peaks
    return float(drawdowns.min())


def compute_metrics_from_values(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {
            "n": 0,
            "expectancy": 0.0,
            "pf": 0.0,
            "max_dd": 0.0,
            "win_rate": 0.0,
        }
    return {
        "n": int(arr.size),
        "expectancy": float(arr.mean()),
        "pf": float(compute_pf(arr)),
        "max_dd": float(compute_max_dd(arr)),
        "win_rate": float((arr > 0.0).mean()),
    }


def support_percentile(reference_values: np.ndarray, observed: float) -> float:
    if reference_values.size == 0:
        return 0.0
    return float(np.mean(reference_values <= observed))


def centrality_from_support(support: float) -> float:
    return max(0.0, 1.0 - 2.0 * abs(float(support) - 0.5))


def upside_discount_weight(n: int) -> float:
    if n <= 1:
        return 0.0
    return clamp_unit((n - 1) / 9.0)


def downside_preservation_weight(n: int) -> float:
    return 1.0


def downside_caution_weight(n: int) -> float:
    if n <= 1:
        return 0.99
    if n == 2:
        return 0.995
    if n == 3:
        return 0.998
    return 1.0


def positive_gain_balance(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    positives = arr[arr > 0.0]
    if positives.size <= 1:
        return 0.0
    gross_positive = float(positives.sum())
    if gross_positive <= 0.0:
        return 0.0
    concentration = float(positives.max()) / gross_positive
    ideal = 1.0 / float(len(positives))
    if ideal >= 1.0:
        return 0.0
    normalized_balance = 1.0 - ((concentration - ideal) / (1.0 - ideal))
    return clamp_unit(normalized_balance)


def confidence_from_raw_support(
    *,
    raw_support_unit: float,
    compatibility_unit: float,
    n: int,
    pnl_values: np.ndarray,
    scoring_version: str,
) -> tuple[float, dict[str, float]]:
    raw_support = clamp_unit(raw_support_unit)
    compatibility = clamp_unit(compatibility_unit)
    diagnostics = {
        "raw_support_unit": round(raw_support, 6),
        "compatibility_unit": round(compatibility, 6),
        "upside_discount_weight": round(upside_discount_weight(n), 6),
        "downside_preservation_weight": round(downside_preservation_weight(n), 6),
        "positive_gain_balance": round(positive_gain_balance(pnl_values), 6),
    }

    if scoring_version == SCORING_VERSION_LEGACY:
        diagnostics["recalibration_branch"] = "LEGACY_DIRECT_SUPPORT"
        diagnostics["confidence_unit"] = round(raw_support, 6)
        return raw_support, diagnostics

    if raw_support >= 0.5:
        upside_excess = (raw_support - 0.5) / 0.5
        compatibility_gate = math.sqrt(compatibility)
        gain_balance = positive_gain_balance(pnl_values)
        gated_excess = upside_excess * compatibility_gate * upside_discount_weight(n) * gain_balance
        confidence_unit = 0.5 + 0.5 * clamp_unit(gated_excess)
        diagnostics.update(
            {
                "recalibration_branch": "UPSIDE_DISCOUNTED",
                "upside_excess_unit": round(upside_excess, 6),
                "compatibility_gate": round(compatibility_gate, 6),
                "gated_excess_unit": round(gated_excess, 6),
                "confidence_unit": round(confidence_unit, 6),
            }
        )
        return confidence_unit, diagnostics

    downside_shortfall = (0.5 - raw_support) / 0.5
    
    if scoring_version == SCORING_VERSION_REFINED:
        weight = downside_caution_weight(n)
        diagnostics["downside_caution_weight"] = round(weight, 6)
    else:
        weight = downside_preservation_weight(n)

    gated_shortfall = downside_shortfall * weight
    confidence_unit = 0.5 - 0.5 * clamp_unit(gated_shortfall)
    diagnostics.update(
        {
            "recalibration_branch": "DOWNSIDE_PRESERVED",
            "downside_shortfall_unit": round(downside_shortfall, 6),
            "gated_shortfall_unit": round(gated_shortfall, 6),
            "confidence_unit": round(confidence_unit, 6),
        }
    )
    return confidence_unit, diagnostics


def low_n_caution_state(n: int) -> str:
    if n < 5:
        return "LT_5"
    if n < 10:
        return "N5_TO_9"
    if n < 20:
        return "N10_TO_19"
    if n < 40:
        return "N20_TO_39"
    return "N40_PLUS"


def confidence_direction(delta_value: float | None) -> str:
    if delta_value is None or abs(delta_value) <= DELTA_FLAT_EPSILON:
        return "FLAT"
    return "UP" if delta_value > 0.0 else "DOWN"


def classify_sequential_state(
    *,
    institutional_confidence_score: float | None,
    n: int,
    low_confidence_streak: int,
    reliable: bool,
    scoring_version: str,
) -> str:
    if not reliable or institutional_confidence_score is None:
        return "SEQUENTIAL_MODEL_NOT_RELIABLE"

    score = float(institutional_confidence_score)
    
    if scoring_version == SCORING_VERSION_LEGACY:
        if score < 1.0 or low_confidence_streak >= 2:
            return "EVIDENCE_MATERIALLY_UNFAVORABLE"
    elif scoring_version == SCORING_VERSION_RECALIBRATED:
        if score < 0.5 or low_confidence_streak >= 3:
            return "EVIDENCE_MATERIALLY_UNFAVORABLE"
    elif scoring_version == SCORING_VERSION_REFINED:
        # Hysteresis: instant catastrophe (<0.1) or persistent deterioration
        is_instant_material = score < 0.1
        is_persistent_material = score < 0.5 and low_confidence_streak >= 3
        # Marginal trim for N10: slightly longer streak for prolonged tension in small samples
        required_tension_streak = 6 if n < 10 else 5
        is_prolonged_tension = score < 5.0 and low_confidence_streak >= required_tension_streak
        if is_instant_material or is_persistent_material or is_prolonged_tension:
            return "EVIDENCE_MATERIALLY_UNFAVORABLE"
    
    if score < 5.0:
        return "EVIDENCE_EARLY_WARNING"
    if score < 20.0:
        return "EVIDENCE_TENSE_BUT_NOT_ALARMING"
    if n < 5:
        return "EVIDENCE_STILL_THIN"
    return "EVIDENCE_ACCUMULATING_NORMALLY"


def recommended_interpretation(
    *,
    state: str,
    institutional_confidence_score: float | None,
    cumulative_compatibility_score: float | None,
) -> str:
    if state == "SEQUENTIAL_MODEL_NOT_RELIABLE":
        return "CHECK_MODEL_RELIABILITY"
    if state == "EVIDENCE_MATERIALLY_UNFAVORABLE":
        return "ESCALATE_TO_TRIBUNAL_NOTE"
    if state in {"EVIDENCE_EARLY_WARNING", "EVIDENCE_TENSE_BUT_NOT_ALARMING"}:
        return "MONITOR_CLOSELY"
    if (
        institutional_confidence_score is not None
        and institutional_confidence_score >= 20.0
        and cumulative_compatibility_score is not None
        and cumulative_compatibility_score < 20.0
    ):
        return "KEEP_ACCUMULATING_NO_OVERREAD"
    return "KEEP_ACCUMULATING"


def quantile_summary(values: np.ndarray) -> dict[str, float] | None:
    if values.size == 0:
        return None
    points = {
        "p01": 0.01,
        "p05": 0.05,
        "p10": 0.10,
        "p25": 0.25,
        "p50": 0.50,
        "p75": 0.75,
        "p90": 0.90,
        "p95": 0.95,
        "p99": 0.99,
    }
    return {label: round(float(np.quantile(values, q)), 4) for label, q in points.items()}


def _parse_time(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series.astype(str), errors="coerce", utc=True)
    return parsed.dt.tz_convert("America/New_York")


def load_historical_trades(line_name: str) -> pd.DataFrame:
    config = LINE_CONFIGS[line_name]
    path = config["historical_path"]
    if not path.exists():
        raise FileNotFoundError(f"Falta baseline historica: {path}")

    raw = pd.read_csv(path)
    pnl_col = config["historical_pnl_col"]
    if pnl_col not in raw.columns:
        raise RuntimeError(f"FAIL-CLOSED: falta columna historica {pnl_col} en {path}")

    frame = pd.DataFrame(
        {
            "source_line": line_name,
            "official_id": [f"{line_name}_HIST_{idx:05d}" for idx in range(len(raw))],
            "session_date": pd.to_datetime(raw["session_date"], errors="coerce").dt.strftime("%Y-%m-%d"),
            "event_time_ny": _parse_time(raw[config["history_time_col"]]),
            "pnl_r": pd.to_numeric(raw[pnl_col], errors="coerce"),
            "level": raw.get("level", pd.Series(index=raw.index, dtype="object")).fillna("").astype(str),
            "direction": raw.get("direction", pd.Series(index=raw.index, dtype="object")).fillna("").astype(str),
            "risk_pips": pd.to_numeric(raw.get("risk_pips", np.nan), errors="coerce"),
            "baseline_source": str(path.relative_to(ROOT)),
        }
    )
    frame = frame.dropna(subset=["pnl_r", "event_time_ny"]).copy()
    frame = frame.sort_values(["event_time_ny", "official_id"]).reset_index(drop=True)
    return frame


def load_forward_trades(line_name: str) -> pd.DataFrame:
    config = LINE_CONFIGS[line_name]
    path = config["forward_path"]
    if not path.exists():
        raise FileNotFoundError(f"Falta forward oficial: {path}")

    raw = pd.read_csv(path)
    if config["forward_kind"] == "GLOBAL":
        official = raw[(raw["event_type"] == "PAPER_EXIT") & raw["pnl_r"].notna()].copy()
        official = official[~official["signal_id"].astype(str).str.startswith(("REHEARSAL_", "DEBUG"))]
        official["official_id"] = official["signal_id"].astype(str)
        official["session_date"] = pd.to_datetime(official["session_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        official["event_time_ny"] = _parse_time(official["exit_time"].fillna(official["entry_time"]))
        official["level"] = official["sweep_level"].fillna("").astype(str)
    else:
        official = raw[raw["event_id"].astype(str).str.startswith("CORE_") & raw["pnl_r"].notna()].copy()
        official = official[~official["event_id"].astype(str).str.startswith(("REHEARSAL_", "DEBUG"))]
        official["official_id"] = official["event_id"].astype(str)
        official["session_date"] = pd.to_datetime(
            official["timestamp_ny"].astype(str).str.slice(0, 10),
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")
        official["event_time_ny"] = _parse_time(official["exit_time"].fillna(official["timestamp_ny"]))
        official["level"] = official["level"].fillna("").astype(str)

    official["source_line"] = line_name
    official["direction"] = official["direction"].fillna("").astype(str)
    official["risk_pips"] = pd.to_numeric(official["risk_pips"], errors="coerce")
    official["pnl_r"] = pd.to_numeric(official["pnl_r"], errors="coerce")
    official = official.dropna(subset=["pnl_r", "event_time_ny"]).copy()
    duplicate_count = int(official["official_id"].duplicated().sum())
    if duplicate_count:
        official = official.sort_values(["event_time_ny", "official_id"]).drop_duplicates("official_id", keep="last")
    official = official.sort_values(["event_time_ny", "official_id"]).reset_index(drop=True)
    return official[
        [
            "source_line",
            "official_id",
            "session_date",
            "event_time_ny",
            "pnl_r",
            "level",
            "direction",
            "risk_pips",
        ]
    ].copy()


def build_reference_model(line_name: str, history: pd.DataFrame | None = None) -> ReferenceModel:
    history_frame = history.copy() if history is not None else load_historical_trades(line_name)
    ordered = history_frame.sort_values(["event_time_ny", "official_id"]).reset_index(drop=True)
    values = ordered["pnl_r"].astype(float).to_numpy()
    max_prefix = min(MAX_PREFIX_CAP, len(ordered) // 4)
    prefix_map: dict[int, PrefixReference] = {}

    for n in range(1, max_prefix + 1):
        starts = np.arange(0, len(values) - n + 1, dtype=int)
        expectancy_values: list[float] = []
        max_dd_values: list[float] = []
        pf_values: list[float] = []
        win_rate_values: list[float] = []

        for start in starts:
            chunk = values[start : start + n]
            metrics = compute_metrics_from_values(chunk)
            expectancy_values.append(metrics["expectancy"])
            max_dd_values.append(metrics["max_dd"])
            pf_values.append(metrics["pf"])
            win_rate_values.append(metrics["win_rate"])

        prefix_map[n] = PrefixReference(
            n=n,
            starts=starts,
            expectancy=np.asarray(expectancy_values, dtype=float),
            max_dd=np.asarray(max_dd_values, dtype=float),
            pf=np.asarray(pf_values, dtype=float),
            win_rate=np.asarray(win_rate_values, dtype=float),
        )

    return ReferenceModel(
        line_name=line_name,
        history=ordered,
        history_hash=sha256_file(LINE_CONFIGS[line_name]["historical_path"]),
        max_prefix=max_prefix,
        prefix_map=prefix_map,
    )


def _active_metric_names(n: int) -> list[str]:
    names = ["expectancy", "max_dd"]
    if n >= 5:
        names.append("pf")
    return names


def _reference_mask(reference: PrefixReference, exclude_interval: tuple[int, int] | None) -> np.ndarray | None:
    if exclude_interval is None:
        return None
    start_idx, end_idx = exclude_interval
    window_end = reference.starts + reference.n - 1
    return (window_end < start_idx) | (reference.starts > end_idx)


def _filtered_metric_values(
    reference: PrefixReference,
    *,
    metric_name: str,
    exclude_interval: tuple[int, int] | None = None,
) -> np.ndarray:
    values = getattr(reference, metric_name)
    mask = _reference_mask(reference, exclude_interval)
    if mask is None:
        return values
    return values[mask]


def score_trade_path(
    line_name: str,
    trades: pd.DataFrame,
    model: ReferenceModel,
    *,
    exclude_interval: tuple[int, int] | None = None,
    scoring_version: str = SCORING_VERSION_RECALIBRATED,
) -> pd.DataFrame:
    ordered = trades.sort_values(["event_time_ny", "official_id"]).reset_index(drop=True).copy()
    rows: list[dict[str, Any]] = []
    previous_confidence: float | None = None
    low_confidence_streak = 0

    for idx in range(len(ordered)):
        prefix = ordered.iloc[: idx + 1].copy()
        n = len(prefix)
        metric_snapshot = compute_metrics_from_values(prefix["pnl_r"].astype(float).to_numpy())
        reliable = n <= model.max_prefix and n in model.prefix_map
        metric_supports: dict[str, float | None] = {}
        metric_centralities: dict[str, float | None] = {}
        metric_reference_counts: dict[str, int] = {}
        metric_reference_quantiles: dict[str, dict[str, float] | None] = {}

        active_metric_names = _active_metric_names(n)
        active_support_values: list[float] = []
        active_centrality_values: list[float] = []

        if reliable:
            reference = model.prefix_map[n]
            for metric_name in active_metric_names:
                filtered = _filtered_metric_values(reference, metric_name=metric_name, exclude_interval=exclude_interval)
                metric_reference_counts[metric_name] = int(filtered.size)
                metric_reference_quantiles[metric_name] = quantile_summary(filtered)
                if filtered.size < MIN_REFERENCE_WINDOWS:
                    reliable = False
                    metric_supports[metric_name] = None
                    metric_centralities[metric_name] = None
                else:
                    support = support_percentile(filtered, metric_snapshot[metric_name])
                    centrality = centrality_from_support(support)
                    metric_supports[metric_name] = round(support, 6)
                    metric_centralities[metric_name] = round(centrality, 6)
                    active_support_values.append(support)
                    active_centrality_values.append(centrality)

        confidence_score = None
        compatibility_score = None
        raw_support_score = None
        recalibration_diagnostics: dict[str, float | str] = {}
        if reliable and active_support_values and active_centrality_values:
            raw_support_unit = geometric_mean_unit(active_support_values)
            compatibility_unit = geometric_mean_unit(active_centrality_values)
            raw_support_score = round(100.0 * raw_support_unit, 4)
            confidence_unit, recalibration_diagnostics = confidence_from_raw_support(
                raw_support_unit=raw_support_unit,
                compatibility_unit=compatibility_unit,
                n=n,
                pnl_values=prefix["pnl_r"].astype(float).to_numpy(),
                scoring_version=scoring_version,
            )
            confidence_score = round(100.0 * confidence_unit, 4)
            compatibility_score = round(100.0 * compatibility_unit, 4)

        delta_trade = None if confidence_score is None or previous_confidence is None else round(confidence_score - previous_confidence, 4)
        if confidence_score is not None and confidence_score < 5.0:
            low_confidence_streak += 1
        else:
            low_confidence_streak = 0
        state = classify_sequential_state(
            institutional_confidence_score=confidence_score,
            n=n,
            low_confidence_streak=low_confidence_streak,
            reliable=reliable,
            scoring_version=scoring_version,
        )
        interpretation = recommended_interpretation(
            state=state,
            institutional_confidence_score=confidence_score,
            cumulative_compatibility_score=compatibility_score,
        )
        if confidence_score is not None:
            previous_confidence = confidence_score

        current_trade = prefix.iloc[-1]
        rows.append(
            {
                "source_line": line_name,
                "official_id": current_trade["official_id"],
                "session_date": current_trade["session_date"],
                "event_time_ny": str(current_trade["event_time_ny"]),
                "trade_pnl_r": round(float(current_trade["pnl_r"]), 4),
                "cumulative_n": n,
                "cumulative_expectancy_r": round(metric_snapshot["expectancy"], 4),
                "cumulative_pf": round(metric_snapshot["pf"], 4),
                "cumulative_max_dd_r": round(metric_snapshot["max_dd"], 4),
                "cumulative_wr": round(metric_snapshot["win_rate"], 4),
                "institutional_confidence_score": confidence_score,
                "raw_support_score": raw_support_score,
                "cumulative_compatibility_score": compatibility_score,
                "sequential_evidence_state": state,
                "evidence_delta_per_trade": delta_trade,
                "low_n_caution_state": low_n_caution_state(n),
                "direction_of_confidence_change": confidence_direction(delta_trade),
                "recommended_interpretation_state": interpretation,
                "active_metric_count": len(active_metric_names),
                "support_expectancy": metric_supports.get("expectancy"),
                "support_max_dd": metric_supports.get("max_dd"),
                "support_pf": metric_supports.get("pf"),
                "centrality_expectancy": metric_centralities.get("expectancy"),
                "centrality_max_dd": metric_centralities.get("max_dd"),
                "centrality_pf": metric_centralities.get("pf"),
                "reference_count_expectancy": metric_reference_counts.get("expectancy", 0),
                "reference_count_max_dd": metric_reference_counts.get("max_dd", 0),
                "reference_count_pf": metric_reference_counts.get("pf", 0),
                "reference_expectancy_quantiles": json.dumps(metric_reference_quantiles.get("expectancy")),
                "reference_max_dd_quantiles": json.dumps(metric_reference_quantiles.get("max_dd")),
                "reference_pf_quantiles": json.dumps(metric_reference_quantiles.get("pf")),
                "scoring_version": scoring_version,
                "recalibration_branch": recalibration_diagnostics.get("recalibration_branch"),
                "upside_discount_weight": recalibration_diagnostics.get("upside_discount_weight"),
                "downside_preservation_weight": recalibration_diagnostics.get("downside_preservation_weight"),
                "positive_gain_balance": recalibration_diagnostics.get("positive_gain_balance"),
                "raw_support_unit": recalibration_diagnostics.get("raw_support_unit"),
                "compatibility_unit": recalibration_diagnostics.get("compatibility_unit"),
                "compatibility_gate": recalibration_diagnostics.get("compatibility_gate"),
                "low_confidence_streak": low_confidence_streak,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame["evidence_delta_per_day"] = np.nan
    frame["day_close_flag"] = False
    grouped = frame.groupby(["source_line", "session_date"], sort=False).tail(1).copy()
    previous_by_line: dict[str, float | None] = {}
    for row_idx, row in grouped.iterrows():
        line_name_row = str(row["source_line"])
        current_score = row["institutional_confidence_score"]
        previous_score = previous_by_line.get(line_name_row)
        delta_day = None if current_score is None or previous_score is None else round(float(current_score) - float(previous_score), 4)
        frame.loc[row_idx, "evidence_delta_per_day"] = delta_day
        frame.loc[row_idx, "day_close_flag"] = True
        if current_score is not None:
            previous_by_line[line_name_row] = float(current_score)
    return frame


def build_daily_snapshot(trace_frame: pd.DataFrame) -> pd.DataFrame:
    if trace_frame.empty:
        return pd.DataFrame()
    daily = trace_frame.groupby(["source_line", "session_date"], sort=False).tail(1).copy()
    daily = daily[
        [
            "source_line",
            "session_date",
            "official_id",
            "cumulative_n",
            "trade_pnl_r",
            "institutional_confidence_score",
            "cumulative_compatibility_score",
            "evidence_delta_per_day",
            "direction_of_confidence_change",
            "sequential_evidence_state",
            "recommended_interpretation_state",
            "low_n_caution_state",
            "cumulative_expectancy_r",
            "cumulative_pf",
            "cumulative_max_dd_r",
            "cumulative_wr",
        ]
    ].copy()
    daily = daily.rename(
        columns={
            "official_id": "last_official_id",
            "trade_pnl_r": "last_trade_pnl_r",
        }
    )
    return daily.reset_index(drop=True)


def expected_next_trade_delta_band(
    line_name: str,
    model: ReferenceModel,
    *,
    current_n: int,
    scoring_version: str = SCORING_VERSION_RECALIBRATED,
) -> dict[str, float] | None:
    if current_n <= 0 or current_n + 1 > model.max_prefix:
        return None

    history = model.history.reset_index(drop=True)
    deltas: list[float] = []
    for start in range(0, len(history) - (current_n + 1) + 1):
        chunk = history.iloc[start : start + current_n + 1].copy()
        path = score_trade_path(line_name, chunk, model, scoring_version=scoring_version)
        if len(path) < current_n + 1:
            continue
        previous_score = path.iloc[current_n - 1]["institutional_confidence_score"]
        next_score = path.iloc[current_n]["institutional_confidence_score"]
        if previous_score is None or next_score is None:
            continue
        deltas.append(float(next_score) - float(previous_score))

    if not deltas:
        return None

    arr = np.asarray(deltas, dtype=float)
    return {
        "p10": round(float(np.quantile(arr, 0.10)), 4),
        "p50": round(float(np.quantile(arr, 0.50)), 4),
        "p90": round(float(np.quantile(arr, 0.90)), 4),
    }


def current_metric_validation(line_name: str, current_path: pd.DataFrame) -> dict[str, Any]:
    if current_path.empty:
        return {"n": 0, "expectancy": 0.0, "pf": 0.0, "max_dd": 0.0, "win_rate": 0.0}
    last_row = current_path.iloc[-1]
    return {
        "n": int(last_row["cumulative_n"]),
        "expectancy": float(last_row["cumulative_expectancy_r"]),
        "pf": float(last_row["cumulative_pf"]),
        "max_dd": float(last_row["cumulative_max_dd_r"]),
        "win_rate": float(last_row["cumulative_wr"]),
    }


def ensure_close(actual: float, expected: float, *, label: str, tol: float = 1e-6) -> None:
    if abs(actual - expected) > tol:
        raise RuntimeError(f"FAIL-CLOSED: mismatch en {label}: actual={actual} expected={expected}")


def validate_against_official_views(
    *,
    line_name: str,
    current_metrics: dict[str, Any],
    scoreboard_df: pd.DataFrame,
    tribunal_map: dict[str, dict[str, Any]],
) -> dict[str, str]:
    score_row = scoreboard_df.loc[scoreboard_df["Line"] == line_name].iloc[0]
    tribunal_row = tribunal_map[line_name]

    ensure_close(current_metrics["n"], as_float(score_row["Sample_N"]), label=f"{line_name}.scoreboard.n")
    ensure_close(current_metrics["expectancy"], as_float(score_row["Exp_Forward"]), label=f"{line_name}.scoreboard.expectancy")
    ensure_close(current_metrics["pf"], as_float(score_row["PF_Forward"]), label=f"{line_name}.scoreboard.pf")
    ensure_close(current_metrics["max_dd"], as_float(score_row["Max_DD_R"]), label=f"{line_name}.scoreboard.max_dd")

    ensure_close(current_metrics["n"], as_float(tribunal_row["n"]), label=f"{line_name}.tribunal.n")
    ensure_close(current_metrics["pf"], as_float(tribunal_row["pf"]), label=f"{line_name}.tribunal.pf")
    ensure_close(current_metrics["max_dd"], as_float(tribunal_row["dd"]), label=f"{line_name}.tribunal.dd")

    return {
        "scoreboard_vs_forward": "PASS",
        "tribunal_vs_forward": "PASS",
    }


def build_line_status_entry(
    *,
    line_name: str,
    model: ReferenceModel,
    forward_trades: pd.DataFrame,
    trace_frame: pd.DataFrame,
    official_validation: dict[str, str],
    validation_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    current_metrics = current_metric_validation(line_name, trace_frame)
    current_row = trace_frame.iloc[-1] if not trace_frame.empty else None
    config = LINE_CONFIGS[line_name]
    notes = [
        config["history_note"],
        config["forward_note"],
        "La capa secuencial es monitor-only y no cambia gating institucional.",
    ]

    line_validation_summary = None
    if validation_summary:
        line_validation_summary = validation_summary.get("lines", {}).get(line_name)

    return {
        "line": line_name,
        "engine": "SEQUENTIAL_FORWARD_EVIDENCE_ENGINE_V1",
        "sequential_model_method": {
            "reference_family": "CONTIGUOUS_PREFIX_WINDOWS",
            "support_score": "GEOMETRIC_MEAN_OF_ONE_SIDED_SUPPORT_PERCENTILES",
            "compatibility_score": "GEOMETRIC_MEAN_OF_BILATERAL_CENTRALITY",
            "active_metrics": ["expectancy", "max_dd", "pf_if_n_gte_5"],
            "min_reference_windows": MIN_REFERENCE_WINDOWS,
            "max_prefix": model.max_prefix,
        },
        "sources": {
            "historical_path": str(config["historical_path"].relative_to(ROOT)),
            "historical_hash": model.history_hash,
            "historical_rows": int(len(model.history)),
            "historical_pnl_field": config["historical_pnl_col"],
            "forward_path": str(config["forward_path"].relative_to(ROOT)),
            "forward_hash": sha256_file(config["forward_path"]),
            "forward_rows": int(len(forward_trades)),
            "trace_path": str(TRACE_CSV.relative_to(ROOT)),
            "daily_path": str(DAILY_CSV.relative_to(ROOT)),
        },
        "current_metrics": {
            "official_sample_n": current_metrics["n"],
            "expectancy_r": safe_round(current_metrics["expectancy"]),
            "pf": safe_round(current_metrics["pf"]),
            "max_dd_r": safe_round(current_metrics["max_dd"]),
            "win_rate": safe_round(current_metrics["win_rate"]),
        },
        "current_state": {
            "scoring_version": None if current_row is None else current_row["scoring_version"],
            "institutional_confidence_score": None if current_row is None else current_row["institutional_confidence_score"],
            "raw_support_score": None if current_row is None else current_row["raw_support_score"],
            "cumulative_compatibility_score": None if current_row is None else current_row["cumulative_compatibility_score"],
            "sequential_evidence_state": "SEQUENTIAL_MODEL_NOT_RELIABLE" if current_row is None else current_row["sequential_evidence_state"],
            "evidence_delta_per_trade": None if current_row is None else current_row["evidence_delta_per_trade"],
            "evidence_delta_per_day": None if current_row is None else current_row["evidence_delta_per_day"],
            "low_n_caution_state": low_n_caution_state(0) if current_row is None else current_row["low_n_caution_state"],
            "direction_of_confidence_change": "FLAT" if current_row is None else current_row["direction_of_confidence_change"],
            "recommended_interpretation_state": "CHECK_MODEL_RELIABILITY" if current_row is None else current_row["recommended_interpretation_state"],
            "support_expectancy": None if current_row is None else current_row["support_expectancy"],
            "support_max_dd": None if current_row is None else current_row["support_max_dd"],
            "support_pf": None if current_row is None else current_row["support_pf"],
            "centrality_expectancy": None if current_row is None else current_row["centrality_expectancy"],
            "centrality_max_dd": None if current_row is None else current_row["centrality_max_dd"],
            "centrality_pf": None if current_row is None else current_row["centrality_pf"],
            "recalibration_branch": None if current_row is None else current_row["recalibration_branch"],
            "upside_discount_weight": None if current_row is None else current_row["upside_discount_weight"],
            "downside_preservation_weight": None if current_row is None else current_row["downside_preservation_weight"],
            "positive_gain_balance": None if current_row is None else current_row["positive_gain_balance"],
            "expected_next_trade_delta_band": expected_next_trade_delta_band(
                line_name,
                model,
                current_n=current_metrics["n"],
                scoring_version=SCORING_VERSION_RECALIBRATED,
            ),
        },
        "integration_posture": {
            "unified_line_status": "ANNOTATION_ONLY",
            "tribunal": "MONITOR_ONLY",
            "scoreboard": "UNCHANGED",
        },
        "validation": {
            "official_view_checks": official_validation,
            "validator_summary": line_validation_summary,
        },
        "notes": notes,
    }


def load_validation_summary() -> dict[str, Any] | None:
    if not VALIDATION_JSON.exists():
        return None
    return read_json(VALIDATION_JSON)
