from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from phase37_ftmo_trial_support import ROOT, OUT, write_json, write_text
from phase37c_mt5_terminal_autodetect import autodetect


PHASE_OUT = OUT.parent / "phase37c_full_auto_ftmo_trial_bootstrap"
SOURCE_DIR = ROOT / "MANIPULANTE" / "09_COMPLIANCE" / "MT5_LIVE_NEWS_ADAPTER"
BRIDGE_SOURCE = SOURCE_DIR / "MANIPULANTE_CalendarBridgeEA.mq5"


EA_TEXT = r'''
#property strict

input int ExportEverySeconds = 600;
input int DaysAhead = 7;

string JsonEscape(string value)
{
   StringReplace(value, "\\", "\\\\");
   StringReplace(value, "\"", "\\\"");
   StringReplace(value, "\r", " ");
   StringReplace(value, "\n", " ");
   return value;
}

string ImportanceText(const ENUM_CALENDAR_EVENT_IMPORTANCE importance)
{
   if(importance == CALENDAR_IMPORTANCE_HIGH) return "HIGH";
   if(importance == CALENDAR_IMPORTANCE_MODERATE) return "MEDIUM";
   if(importance == CALENDAR_IMPORTANCE_LOW) return "LOW";
   return "UNKNOWN";
}

bool TargetCurrency(const string currency)
{
   return (currency == "EUR" || currency == "USD");
}

string UtcText(datetime value)
{
   return TimeToString(value, TIME_DATE | TIME_SECONDS) + "Z";
}

void ExportRange(datetime from_time, datetime to_time, string file_name)
{
   MqlCalendarValue values[];
   int count = CalendarValueHistory(values, from_time, to_time, NULL, NULL);
   int handle = FileOpen("MANIPULANTE\\" + file_name, FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(handle == INVALID_HANDLE)
   {
      Print("MANIPULANTE CalendarBridge cannot open ", file_name);
      return;
   }
   FileWriteString(handle, "{\n");
   FileWriteString(handle, "  \"source_type\": \"MT5_MQL5_CALENDAR\",\n");
   FileWriteString(handle, "  \"verified_by_mt5\": true,\n");
   FileWriteString(handle, "  \"generated_at_utc\": \"" + UtcText(TimeGMT()) + "\",\n");
   FileWriteString(handle, "  \"server_time\": \"" + TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS) + "\",\n");
   FileWriteString(handle, "  \"timezone_basis\": \"TimeGMT plus MT5 server TimeCurrent\",\n");
   FileWriteString(handle, "  \"events\": [\n");
   bool first = true;
   for(int i = 0; i < count; i++)
   {
      MqlCalendarEvent event;
      MqlCalendarCountry country;
      if(!CalendarEventById(values[i].event_id, event)) continue;
      if(!CalendarCountryById(event.country_id, country)) continue;
      if(!TargetCurrency(country.currency)) continue;
      if(event.importance != CALENDAR_IMPORTANCE_HIGH) continue;
      if(!first) FileWriteString(handle, ",\n");
      first = false;
      FileWriteString(handle, "    {\n");
      FileWriteString(handle, "      \"event_id\": \"" + IntegerToString((int)values[i].event_id) + "\",\n");
      FileWriteString(handle, "      \"event_name\": \"" + JsonEscape(event.name) + "\",\n");
      FileWriteString(handle, "      \"currency\": \"" + country.currency + "\",\n");
      FileWriteString(handle, "      \"impact\": \"" + ImportanceText(event.importance) + "\",\n");
      FileWriteString(handle, "      \"event_time_utc\": \"" + UtcText(values[i].time) + "\",\n");
      FileWriteString(handle, "      \"event_time_server\": \"" + TimeToString(values[i].time, TIME_DATE | TIME_SECONDS) + "\",\n");
      FileWriteString(handle, "      \"source\": \"MT5_MQL5_CALENDAR\"\n");
      FileWriteString(handle, "    }");
   }
   FileWriteString(handle, "\n  ]\n");
   FileWriteString(handle, "}\n");
   FileClose(handle);
}

void ExportNews()
{
   datetime now_utc = TimeGMT();
   MqlDateTime dt;
   TimeToStruct(now_utc, dt);
   dt.hour = 0;
   dt.min = 0;
   dt.sec = 0;
   datetime start_today = StructToTime(dt);
   datetime end_today = start_today + 86400;
   datetime end_week = start_today + (DaysAhead * 86400);
   ExportRange(start_today, end_today, "ftmo_news_today.json");
   ExportRange(start_today, end_week, "ftmo_news_week.json");
   ExportRange(start_today, end_week, "ftmo_news_gate_status.json");
}

int OnInit()
{
   EventSetTimer(MathMax(300, ExportEverySeconds));
   ExportNews();
   return INIT_SUCCEEDED;
}

void OnTimer()
{
   ExportNews();
}

void OnDeinit(const int reason)
{
   EventKillTimer();
}
'''


def find_metaeditor(terminal_path: str | None) -> Path | None:
    candidates: list[Path] = []
    if terminal_path:
        candidates.extend([Path(terminal_path) / "MetaEditor64.exe", Path(terminal_path) / "MetaEditor.exe"])
    candidates.extend([Path(r"C:\Program Files\MetaTrader 5\MetaEditor64.exe"), Path(r"C:\Program Files\MetaTrader 5\MetaEditor.exe")])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def static_trade_scan(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    patterns = ["OrderSend", "CTrade", r"\bBuy\s*\(", r"\bSell\s*\(", "PositionOpen", r"\btrade\."]
    return [pattern for pattern in patterns if re.search(pattern, text, re.I)]


def install_bridge() -> dict[str, Any]:
    auto = autodetect()
    status: dict[str, Any] = {
        "timestamp_utc": auto.get("timestamp_utc"),
        "state": "BLOCKED_NO_DATA_PATH",
        "data_path": auto.get("data_path"),
        "terminal_path": auto.get("terminal_path"),
        "experts_dir": None,
        "files_dir": None,
        "source_path": str(BRIDGE_SOURCE),
        "installed_path": None,
        "metaeditor": None,
        "compile_returncode": None,
        "compile_log_path": str(PHASE_OUT / "calendar_bridge_install" / "phase37c_calendar_bridge_compile.log"),
        "ex5_exists": False,
        "trade_findings": [],
        "reason": "",
    }
    if auto.get("state") != "FTMO_MT5_DEMO_AUTODETECTED":
        status["state"] = auto.get("state", "BLOCKED_NO_DATA_PATH")
        status["reason"] = auto.get("reason")
        return status
    data_path = Path(str(auto["data_path"]))
    experts_dir = data_path / "MQL5" / "Experts" / "MANIPULANTE"
    files_dir = data_path / "MQL5" / "Files" / "MANIPULANTE"
    try:
        experts_dir.mkdir(parents=True, exist_ok=True)
        files_dir.mkdir(parents=True, exist_ok=True)
        SOURCE_DIR.mkdir(parents=True, exist_ok=True)
        BRIDGE_SOURCE.write_text(EA_TEXT.strip() + "\n", encoding="utf-8")
        installed = experts_dir / "MANIPULANTE_CalendarBridgeEA.mq5"
        shutil.copy2(BRIDGE_SOURCE, installed)
    except Exception as exc:
        status["state"] = "BLOCKED_CANNOT_WRITE_MQL5_PATH"
        status["reason"] = str(exc)
        return status
    status["experts_dir"] = str(experts_dir)
    status["files_dir"] = str(files_dir)
    status["installed_path"] = str(installed)
    findings = static_trade_scan(installed)
    status["trade_findings"] = findings
    if findings:
        status["state"] = "BLOCKED_TRADE_FUNCTION_FOUND"
        status["reason"] = "Static scan found trade-like token"
        return status
    metaeditor = find_metaeditor(auto.get("terminal_path"))
    status["metaeditor"] = str(metaeditor) if metaeditor else None
    if metaeditor is None:
        status["state"] = "BLOCKED_METAEDITOR_NOT_FOUND"
        status["reason"] = "MetaEditor executable not found"
        return status
    log_path = PHASE_OUT / "calendar_bridge_install" / "phase37c_calendar_bridge_compile.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(metaeditor), f"/compile:{installed}", f"/log:{log_path}"]
    completed = subprocess.run(cmd, cwd=experts_dir, capture_output=True, text=True, timeout=120)
    status["compile_returncode"] = completed.returncode
    if completed.stdout or completed.stderr:
        log_path.write_text((completed.stdout or "") + "\n" + (completed.stderr or ""), encoding="utf-8")
    elif not log_path.exists():
        log_path.write_text("MetaEditor returned no stdout/stderr. Check .ex5 existence and terminal journal if needed.\n", encoding="utf-8")
    ex5 = installed.with_suffix(".ex5")
    status["ex5_exists"] = ex5.exists()
    if not ex5.exists():
        status["state"] = "BLOCKED_COMPILE_FAILED"
        status["reason"] = "MetaEditor did not produce .ex5"
        return status
    status["state"] = "CALENDAR_BRIDGE_COMPILED"
    status["reason"] = "Bridge installed and .ex5 produced; MetaEditor return code recorded for audit"
    return status


def write_outputs() -> dict[str, Any]:
    status = install_bridge()
    write_json(PHASE_OUT / "calendar_bridge_install" / "phase37c_calendar_bridge_install.json", status)
    write_text(
        PHASE_OUT / "calendar_bridge_install" / "phase37c_calendar_bridge_install.md",
        f"""
# Phase37C Calendar Bridge Install

- state: {status['state']}
- data_path: {status['data_path']}
- installed_path: {status['installed_path']}
- metaeditor: {status['metaeditor']}
- compiled: {status['ex5_exists']}
- trade findings: {status['trade_findings']}
- reason: {status['reason']}
""",
    )
    return status


if __name__ == "__main__":
    print(json.dumps(write_outputs(), indent=2, ensure_ascii=False))
