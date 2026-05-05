from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from phase36_exness_symbol_gate import get_exness_symbol_gate_status


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "BOT_V2_DAYTIME_LAB" / "outputs" / "phase36s_live_news_lot_feasibility" / "lot_feasibility_100usd"


@dataclass(frozen=True)
class FeasibilityRow:
    balance_usd: float
    risk_pct: float
    risk_label: str
    stop_pips: float
    min_lot: float
    lot_step: float
    pip_value_per_lot_usd: float
    theoretical_lot: float
    rounded_lot: float
    nominal_risk_usd: float
    actual_risk_usd: float
    actual_risk_pct: float
    exceeds_risk: bool
    allowed_technical: bool
    allowed_real_today: bool
    reason: str


def round_down(value: float, step: float) -> float:
    if step <= 0:
        return 0.0
    return math.floor(value / step) * step


def build_rows(
    balance: float = 100.0,
    min_lot: float = 0.01,
    lot_step: float = 0.01,
    spread_pips: float = 0.8,
    pip_value_per_lot_usd: float = 10.0,
) -> list[FeasibilityRow]:
    risks = [0.001, 0.0025, 0.005, 0.0075, 0.01]
    stops = [3, 5, 8, 10, 15, 20, 25, 30]
    rows: list[FeasibilityRow] = []
    for risk in risks:
        nominal_risk = balance * risk
        for stop in stops:
            theoretical_lot = nominal_risk / (stop * pip_value_per_lot_usd)
            rounded = round_down(theoretical_lot, lot_step)
            if rounded < min_lot:
                rounded = min_lot
            actual_risk = rounded * stop * pip_value_per_lot_usd
            actual_pct = actual_risk / balance
            exceeds = actual_pct > risk + 1e-12
            spread_ratio = spread_pips / stop if stop else 1.0
            reason = "ALLOW_TECHNICAL"
            technical = True
            if spread_ratio > 0.12:
                technical = False
                reason = "STOP_TOO_SMALL_VS_SPREAD"
            elif exceeds:
                technical = False
                reason = "MIN_LOT_EXCEEDS_RISK"
            real_today = technical and risk <= 0.0025
            if risk >= 0.0075:
                real_today = False
                if technical:
                    reason = "RISK_NOT_AUTHORIZED_TODAY"
            elif risk == 0.005:
                real_today = False
                if technical:
                    reason = "RISK_050_DRY_RUN_ONLY"
            rows.append(
                FeasibilityRow(
                    balance_usd=balance,
                    risk_pct=risk,
                    risk_label=f"{risk*100:.2f}%",
                    stop_pips=stop,
                    min_lot=min_lot,
                    lot_step=lot_step,
                    pip_value_per_lot_usd=pip_value_per_lot_usd,
                    theoretical_lot=round(theoretical_lot, 5),
                    rounded_lot=round(rounded, 2),
                    nominal_risk_usd=round(nominal_risk, 4),
                    actual_risk_usd=round(actual_risk, 4),
                    actual_risk_pct=round(actual_pct, 6),
                    exceeds_risk=exceeds,
                    allowed_technical=technical,
                    allowed_real_today=real_today,
                    reason=reason,
                )
            )
    return rows


def summarize(rows: list[FeasibilityRow], spread_pips: float) -> dict[str, Any]:
    def allowed(risk: float) -> bool:
        return any(row.risk_pct == risk and row.allowed_real_today for row in rows)

    feasible_stops = sorted({row.stop_pips for row in rows if (spread_pips / row.stop_pips) <= 0.12})
    reference_stop = feasible_stops[0] if feasible_stops else 8
    min_lot = rows[0].min_lot if rows else 0.01
    pip_value = rows[0].pip_value_per_lot_usd if rows else 10.0
    min_risk_usd_ref = min_lot * reference_stop * pip_value
    min_risk_pct_ref = min_risk_usd_ref / 100.0
    balance_min_025 = min_risk_usd_ref / 0.0025
    balance_min_050 = min_risk_usd_ref / 0.005
    return {
        "balance_usd": 100.0,
        "symbol": "EURUSDm",
        "min_lot": min_lot,
        "lot_step": rows[0].lot_step if rows else 0.01,
        "spread_pips": spread_pips,
        "pip_value_per_lot_usd": pip_value,
        "pip_value_source": "conservative EURUSD standard lot approximation",
        "reference_stop_pips": reference_stop,
        "minimum_real_risk_usd_at_reference_stop": round(min_risk_usd_ref, 4),
        "minimum_real_risk_pct_at_reference_stop": round(min_risk_pct_ref, 6),
        "risk_010_allowed": allowed(0.001),
        "risk_025_allowed": allowed(0.0025),
        "risk_050_allowed": False,
        "risk_075_allowed": False,
        "risk_100_allowed": False,
        "balance_min_for_025_at_reference_stop": round(balance_min_025, 2),
        "balance_min_for_050_at_reference_stop": round(balance_min_050, 2),
        "cent_or_micro_account_helpful": True,
        "micro_real_blocked_by_lot": not (allowed(0.001) or allowed(0.0025)),
    }


def write_outputs() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    symbol = get_exness_symbol_gate_status()
    min_lot = float(symbol.get("min_lot") or 0.01)
    lot_step = float(symbol.get("lot_step") or 0.01)
    spread_pips = float(symbol.get("spread_pips") or 0.8)
    rows = build_rows(min_lot=min_lot, lot_step=lot_step, spread_pips=spread_pips)
    row_dicts = [asdict(row) for row in rows]
    with (OUT / "phase36s_lot_feasibility_100usd.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row_dicts[0].keys()))
        writer.writeheader()
        writer.writerows(row_dicts)
    summary = summarize(rows, spread_pips)
    summary["symbol_gate"] = symbol
    (OUT / "phase36s_lot_feasibility_100usd.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = [
        "# Phase36S Lot Feasibility 100 USD",
        "",
        f"- min_lot: {summary['min_lot']}",
        f"- lot_step: {summary['lot_step']}",
        f"- spread_pips: {summary['spread_pips']}",
        f"- reference_stop_pips: {summary['reference_stop_pips']}",
        f"- minimum_real_risk_pct_at_reference_stop: {summary['minimum_real_risk_pct_at_reference_stop']:.4%}",
        f"- 0.10 allowed: {summary['risk_010_allowed']}",
        f"- 0.25 allowed: {summary['risk_025_allowed']}",
        f"- 0.50 allowed today: {summary['risk_050_allowed']}",
        f"- balance_min_for_025: {summary['balance_min_for_025_at_reference_stop']}",
        f"- balance_min_for_050: {summary['balance_min_for_050_at_reference_stop']}",
        f"- micro_real_blocked_by_lot: {summary['micro_real_blocked_by_lot']}",
    ]
    (OUT / "phase36s_lot_feasibility_100usd.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(write_outputs(), indent=2))
