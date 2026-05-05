from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "BOT_V2_DAYTIME_LAB" / "outputs" / "phase36_live_news_mt5_dryrun" / "lot_validation"


@dataclass(frozen=True)
class LotScenario:
    balance_usd: float
    risk_pct: float
    stop_pips: float
    risk_usd: float
    raw_lot: float
    rounded_lot: float
    actual_risk_usd: float
    actual_risk_pct: float
    status: str
    reason: str


class ExnessLotValidator:
    """Micro-real lot validator used only for future activation planning.

    It never connects to MT5 and never sends orders. Lot size is rounded down.
    If the minimum lot would exceed the requested risk, the scenario blocks.
    """

    def __init__(
        self,
        min_lot: float = 0.01,
        lot_step: float = 0.01,
        pip_value_per_lot_usd: float = 10.0,
        max_spread_sl_ratio: float = 0.12,
        assumed_spread_pips: float = 0.7,
    ) -> None:
        self.min_lot = min_lot
        self.lot_step = lot_step
        self.pip_value_per_lot_usd = pip_value_per_lot_usd
        self.max_spread_sl_ratio = max_spread_sl_ratio
        self.assumed_spread_pips = assumed_spread_pips

    def validate(self, balance_usd: float, risk_pct: float, stop_pips: float) -> LotScenario:
        risk_usd = balance_usd * risk_pct
        if risk_pct >= 0.01:
            return self._blocked(balance_usd, risk_pct, stop_pips, risk_usd, "RISK_1PCT_PROHIBITED")
        if risk_pct >= 0.0075:
            return self._blocked(balance_usd, risk_pct, stop_pips, risk_usd, "RISK_075_BLOCKED_FOR_INITIAL_MICRO_REAL")
        if stop_pips < 5:
            return self._blocked(balance_usd, risk_pct, stop_pips, risk_usd, "STOP_TOO_SMALL")
        if self.assumed_spread_pips / stop_pips > self.max_spread_sl_ratio:
            return self._blocked(balance_usd, risk_pct, stop_pips, risk_usd, "SPREAD_SL_RATIO_TOO_HIGH")
        raw_lot = risk_usd / (stop_pips * self.pip_value_per_lot_usd)
        rounded_lot = self._round_down(raw_lot)
        if rounded_lot < self.min_lot:
            min_lot_risk = stop_pips * self.pip_value_per_lot_usd * self.min_lot
            if min_lot_risk > risk_usd:
                return LotScenario(
                    balance_usd=balance_usd,
                    risk_pct=risk_pct,
                    stop_pips=stop_pips,
                    risk_usd=round(risk_usd, 4),
                    raw_lot=round(raw_lot, 5),
                    rounded_lot=0.0,
                    actual_risk_usd=0.0,
                    actual_risk_pct=0.0,
                    status="BLOCK",
                    reason="MIN_LOT_EXCEEDS_RISK",
                )
            rounded_lot = self.min_lot
        actual_risk_usd = rounded_lot * stop_pips * self.pip_value_per_lot_usd
        actual_risk_pct = actual_risk_usd / balance_usd
        status = "ALLOW_DRY_RUN_VALIDATION_ONLY"
        reason = "VALID_FOR_DRY_RUN_NOT_AUTHORIZED_FOR_REAL"
        return LotScenario(
            balance_usd=balance_usd,
            risk_pct=risk_pct,
            stop_pips=stop_pips,
            risk_usd=round(risk_usd, 4),
            raw_lot=round(raw_lot, 5),
            rounded_lot=round(rounded_lot, 2),
            actual_risk_usd=round(actual_risk_usd, 4),
            actual_risk_pct=round(actual_risk_pct, 6),
            status=status,
            reason=reason,
        )

    def _blocked(self, balance: float, risk_pct: float, stop_pips: float, risk_usd: float, reason: str) -> LotScenario:
        raw_lot = risk_usd / (stop_pips * self.pip_value_per_lot_usd)
        return LotScenario(
            balance_usd=balance,
            risk_pct=risk_pct,
            stop_pips=stop_pips,
            risk_usd=round(risk_usd, 4),
            raw_lot=round(raw_lot, 5),
            rounded_lot=0.0,
            actual_risk_usd=0.0,
            actual_risk_pct=0.0,
            status="BLOCK",
            reason=reason,
        )

    def _round_down(self, raw_lot: float) -> float:
        steps = int(raw_lot / self.lot_step)
        return steps * self.lot_step


def build_default_scenarios() -> list[LotScenario]:
    validator = ExnessLotValidator()
    balances = [50, 100, 200, 500]
    risks = [0.001, 0.0025, 0.005, 0.0075, 0.01]
    stops = [3, 5, 8, 10, 15, 20]
    return [validator.validate(balance, risk, stop) for balance in balances for risk in risks for stop in stops]


def write_outputs(scenarios: Iterable[LotScenario] | None = None) -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [asdict(item) for item in (scenarios or build_default_scenarios())]
    csv_path = OUTPUT_DIR / "phase36_lot_scenarios.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    balance_100 = [row for row in rows if row["balance_usd"] == 100]
    summary = {
        "scenario_count": len(rows),
        "balance_100_validated": True,
        "recommended_risk": "0.10% to 0.25% for future micro-real planning; 0.50% dry-run only until Phase37",
        "risk_075_authorized": False,
        "risk_100_authorized": False,
        "blocked_count": sum(1 for row in rows if row["status"] == "BLOCK"),
        "allowed_dry_run_count": sum(1 for row in rows if row["status"] != "BLOCK"),
        "balance_100_allowed_rows": sum(1 for row in balance_100 if row["status"] != "BLOCK"),
    }
    (OUTPUT_DIR / "phase36_lot_validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = [
        "# Phase36 Lot Validation",
        "",
        f"- scenario_count: {summary['scenario_count']}",
        f"- balance_100_validated: {summary['balance_100_validated']}",
        f"- recommended_risk: {summary['recommended_risk']}",
        f"- 0.75_authorized: {summary['risk_075_authorized']}",
        f"- 1.00_authorized: {summary['risk_100_authorized']}",
        "",
        "All allowed rows are validation-only and do not authorize real orders in Phase36.",
    ]
    (OUTPUT_DIR / "phase36_lot_validation.md").write_text("\n".join(md), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(write_outputs(), indent=2))
