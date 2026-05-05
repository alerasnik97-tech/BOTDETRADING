from __future__ import annotations

import csv
import json
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from phase36_live_news_fortress import LiveNewsFortress


ROOT = Path(__file__).resolve().parents[2]
MANIPULANTE_CONFIG = ROOT / "MANIPULANTE" / "01_ESTRATEGIA_AUTORIDAD" / "manipulante_config.json"
WATCH_ONLY_CONFIG = ROOT / "MANIPULANTE" / "03_MT5_DEMO_LAUNCHER" / "MANIPULANTE_WATCH_ONLY_CONFIG.json"
LOG_DIR = ROOT / "MANIPULANTE" / "10_LOGS_PAPER" / "dry_run_decisions"
NY = ZoneInfo("America/New_York")


class ManipulanteDryRunEngine:
    """Watch-only decision engine.

    This class simulates the full gate flow and writes an audit row. It has no
    import dependency on MetaTrader5 and no order_send capability.
    """

    def __init__(self) -> None:
        self.manipulante_config = self._read_json(MANIPULANTE_CONFIG)
        self.watch_only_config = self._read_json(WATCH_ONLY_CONFIG, default={})
        self.news_gate = LiveNewsFortress()
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    def _read_json(self, path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
        if not path.exists():
            if default is not None:
                return default
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def _time_gate(self, now_ny: datetime) -> tuple[str, str]:
        if now_ny.weekday() >= 5:
            return "NO_TRADE_TIME_WINDOW", "weekend"
        start = time.fromisoformat(self.manipulante_config.get("window_start_ny", "07:00"))
        end = time.fromisoformat(self.manipulante_config.get("window_end_ny", "16:30"))
        if not (start <= now_ny.time() <= end):
            return "NO_TRADE_TIME_WINDOW", "outside 07:00-16:30 NY"
        return "ALLOW", "inside window"

    def _weekend_gate(self, now_ny: datetime) -> tuple[str, str]:
        policy = self.manipulante_config.get("global_weekend_policy", {})
        hard_close = str(policy.get("hard_close_time_ny", "16:55"))
        if now_ny.weekday() == 4:
            close_time = time.fromisoformat(hard_close)
            if now_ny.time() >= close_time:
                return "NO_TRADE_WEEKEND_POLICY", f"friday hard close reached {hard_close} NY"
        return "ALLOW", "weekend policy ok"

    def _data_mask_gate(self) -> tuple[str, str]:
        # Phase36 is a live dry-run layer. Without a current live data-mask
        # adapter, the dry-run can still log a blocked decision, but it cannot
        # authorize real trading.
        return "NO_TRADE_DATA_MASK", "live Data Quality Mask adapter not connected in Phase36"

    def _lot_gate(self) -> tuple[str, str, float | None]:
        if not self.watch_only_config.get("lot_validator_required", True):
            return "NO_TRADE_LOT_VALIDATION", "lot validator required flag missing", None
        return "ALLOW", "lot validator available for dry-run only", 0.0

    def run_once(self, now: datetime | None = None) -> dict[str, Any]:
        now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        now_ny = now_utc.astimezone(NY)
        news_status = self.news_gate.get_news_gate_status()
        news_gate = news_status.get("gate", "NO_TRADE")
        data_mask_gate, data_reason = self._data_mask_gate()
        time_gate, time_reason = self._time_gate(now_ny)
        weekend_gate, weekend_reason = self._weekend_gate(now_ny)
        lot_status, lot_reason, simulated_lot = self._lot_gate()
        signal_status = "DRY_RUN_NO_SIGNAL"
        final_decision = "DRY_RUN_NO_SIGNAL"
        reason = "no live signal module connected; order_sent=false"

        if news_gate != "ALLOW":
            final_decision = "NO_TRADE_NEWS_BLOCK"
            reason = str(news_status.get("status", "news gate not allow"))
        elif data_mask_gate != "ALLOW":
            final_decision = "NO_TRADE_DATA_MASK"
            reason = data_reason
        elif time_gate != "ALLOW":
            final_decision = "NO_TRADE_TIME_WINDOW"
            reason = time_reason
        elif weekend_gate != "ALLOW":
            final_decision = "NO_TRADE_WEEKEND_POLICY"
            reason = weekend_reason
        elif lot_status != "ALLOW":
            final_decision = "NO_TRADE_LOT_VALIDATION"
            reason = lot_reason

        row = {
            "timestamp": now_utc.isoformat(),
            "NY_time": now_ny.isoformat(),
            "symbol": self.manipulante_config.get("symbol", "EURUSD"),
            "news_gate": news_gate,
            "data_mask_gate": data_mask_gate,
            "time_gate": time_gate,
            "weekend_gate": weekend_gate,
            "signal_status": signal_status,
            "lot_status": lot_status,
            "final_decision": final_decision,
            "reason": reason,
            "simulated_entry": "",
            "simulated_sl": "",
            "simulated_tp": "",
            "simulated_lot": simulated_lot if simulated_lot is not None else "",
            "order_sent": False,
        }
        self.write_decision(row)
        return row

    def write_decision(self, row: dict[str, Any]) -> Path:
        day = datetime.now(NY).strftime("%Y-%m-%d")
        path = LOG_DIR / f"{day}_decisions.csv"
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        return path


if __name__ == "__main__":
    print(json.dumps(ManipulanteDryRunEngine().run_once(), indent=2))
