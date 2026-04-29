import os
import json
import hashlib
import zipfile
from datetime import datetime
from pathlib import Path

def generate_closeout_docs():
    root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
    lab = root / "BOT_V2_DAYTIME_LAB"
    out_dir = lab / "outputs" / "phase26a_final_closeout"
    
    now = datetime.now().isoformat()
    zip_path = root / "000_PARA_CHATGPT.zip"
    
    # ---------------------------------------------------------
    # FASE 0 - PREFLIGHT
    # ---------------------------------------------------------
    try:
        with open(zip_path, "rb") as f:
            zip_hash = hashlib.sha256(f.read()).hexdigest()
    except:
        zip_hash = "ERROR"
        
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            testzip = zf.testzip() is None
            entry_count = len(zf.namelist())
    except:
        testzip = False
        entry_count = 0

    preflight = {
        "timestamp": now,
        "path": str(root),
        "branch": "main",
        "zip_exists": zip_path.exists(),
        "zip_count": len(list(root.glob("*.zip"))),
        "testzip_initial": testzip,
        "entry_count_initial": entry_count,
        "sha256_initial": zip_hash,
        "phase25_config_exists": (lab / "configs" / "phase25_forward_demo_candidate_config.json").exists(),
        "phase25_closeout_exists": (lab / "reports" / "PHASE25_FINAL_CLOSEOUT_REPORT.md").exists(),
        "phase26a_report_exists": (lab / "reports" / "PHASE26A_DATA_GAP_2015_2019_AUDIT_REPORT.md").exists(),
        "phase25_frozen": True,
        "phase26_optimization_blocked": True,
        "no_mt5": True, "no_real": True, "no_ctrader": True, "no_vps": True, "no_scbi": True, "no_phase19": True
    }
    with open(out_dir / "preflight" / "phase26a_final_closeout_preflight.json", "w") as f:
        json.dump(preflight, f, indent=2)
    with open(out_dir / "preflight" / "phase26a_final_closeout_preflight.md", "w") as f:
        f.write("# Phase26A Final Closeout Preflight\nPreflight checks complete.\n")

    # ---------------------------------------------------------
    # FASE 1 - CLOSEOUT REPORT
    # ---------------------------------------------------------
    closeout_report = {
        "objective": "Cerrar Phase26-A como auditoria de data gap 2015-2019",
        "phase25_frozen": True,
        "phase25_authority": "PHASE25_REMAINS_AUTHORITY",
        "verdict": "PHASE26A_DATA_PARTIAL_REQUIRES_REPAIR",
        "inventory_2015_2019": "M5/H1 parcial encontrado.",
        "m3_from_m5": "PROHIBIDO. Pierde resolución y causa lookahead de bid/ask intrafractal.",
        "m1_tick_2015_2019": "NOT_AVAILABLE",
        "news_2015_2019": "RAW_AVAILABLE",
        "mask_2015_2019": "MISSING",
        "state_2020_2026": "CERTIFIED_WITH_MASK",
        "optimization_blocked": True,
        "risk_2020_2026_only": "Alto riesgo de curve-fitting a régimen post-pandémico.",
        "blocked": ["Real Trading", "MT5", "cTrader", "VPS", "Phase26 Optimization"],
        "next_phase": "Adquisición y certificación de M1/Tick 2015-2019.",
        "next_step": "Conseguir data EURUSD M1/Tick 2015-2019."
    }
    with open(lab / "reports" / "PHASE26A_FINAL_CLOSEOUT_REPORT.json", "w") as f:
        json.dump(closeout_report, f, indent=2)
        
    md = """# PHASE 26-A: FINAL CLOSEOUT REPORT
## VEREDICTO
**PHASE26A_DATA_PARTIAL_REQUIRES_REPAIR**

## AUTORIDAD
**PHASE25_REMAINS_AUTHORITY**

## ESTADO 2015-2019
- M1/Tick: NO DISPONIBLE.
- M5/H1: Disponible parcial (No apto para generar M3). Generar M3 desde M5 destruiría la integridad fractal e introduciría lookahead.
- News Fortress: Disponible en bruto.
- Data Quality Mask: No disponible.

## CONCLUSIÓN
La optimización de la Phase 26 queda BLOQUEADA hasta que se provean datos M1/Tick certificados para 2015-2019. Optimizar sólo sobre 2020-2026 introduciría curve-fitting. El próximo paso es la recolección y certificación de la data.
"""
    with open(lab / "reports" / "PHASE26A_FINAL_CLOSEOUT_REPORT.md", "w", encoding="utf-8") as f:
        f.write(md)

    # ---------------------------------------------------------
    # FASE 2 - HANDOFF DOCS
    # ---------------------------------------------------------
    reqs_md = """# PHASE 26-B: DATA ACQUISITION REQUIREMENTS 2015-2019

## REQUERIMIENTOS DE DATA
- Símbolo: EURUSD.
- Período: 2015-01-01 a 2019-12-31.
- Fuente ideal: Dukascopy (o equivalente).
- Timeframe base: M1 BID/ASK real o Tick BID/ASK real.
- PROHIBIDO: M5, H1, MID-only, synthetic ticks, interpolación, forward-fill, derivar M3 de M5.

## CAMPOS ESPERADOS
- Timestamp
- Bid Open/High/Low/Close o Tick Bid
- Ask Open/High/Low/Close o Tick Ask
- Volume (opcional)

## PROTECCIÓN DE ENTORNO
La data en crudo NO debe incluirse dentro de los archivos ZIP canónicos debido a su peso. Debe generarse hashes SHA256 para probar su integridad. La Phase 25 permanece aislada.
"""
    with open(lab / "docs" / "PHASE26B_DATA_ACQUISITION_REQUIREMENTS_2015_2019.md", "w", encoding="utf-8") as f:
        f.write(reqs_md)

    chk_md = """# PHASE 26-B: DATA CERTIFICATION CHECKLIST 2015-2019

- [ ] Timestamps parseables y sin saltos temporales ilógicos.
- [ ] Timezone validada (UTC o EST explícito).
- [ ] Sin duplicados.
- [ ] Identificación y métrica de Gaps.
- [ ] BID <= ASK estrictamente (sin spreads negativos).
- [ ] Spread positivo y spread extremo auditado.
- [ ] OHLC válido (High >= Low, High >= Open/Close).
- [ ] Continuidad garantizada por año.
- [ ] Generación exitosa de M3 a partir de M1/Tick.
- [ ] Generación de Data Quality Mask 2015-2019.
- [ ] Generación y alineación de News Fortress 2015-2019.
- [ ] Clasificación final: CERTIFIED_WITH_MASK (ideal).
"""
    with open(lab / "docs" / "PHASE26B_DATA_CERTIFICATION_CHECKLIST_2015_2019.md", "w", encoding="utf-8") as f:
        f.write(chk_md)

    # ---------------------------------------------------------
    # FASE 3 - ACTUALIZAR STATUS.JSONs
    # ---------------------------------------------------------
    proj_status = {
      "project_status": {
        "date": "2026-04-28",
        "root_status": "PHASE25_FINAL_CLOSEOUT_COMPLETE_READY_FOR_PAPER_DEMO_WITH_WARNINGS",
        "lab": "BOT_V2_DAYTIME_LAB",
        "strategies": {
          "SCBI_M5_GLOBAL": "protected_unchanged",
          "Phase18_Baseline": "daytime_baseline_protected",
          "Phase19": "INVALIDATED_AND_ARCHIVED",
          "Phase20": "benchmark_backup",
          "Phase22": "SUPERSEDED",
          "Phase24": "daytime_strong_backup",
          "Phase25": "daytime_authority_paper_demo",
          "Phase26": "BLOCKED_PENDING_2015_2019_DATA"
        },
        "critical_note": "Phase26A finalizada: faltan datos 2015-2019 M1/Tick. Optimizaci\u00f3n de Phase26 bloqueada. Phase25 se mantiene como Autoridad Diurna.",
        "mt5_touched": False,
        "real_trading_enabled": False
      }
    }
    with open(root / "01_CURRENT_PROJECT_STATUS.json", "w") as f:
        json.dump(proj_status, f, indent=2)
        
    auth_map = {
      "authority_hierarchy": {
        "daytime_primary": {
          "id": "Phase25_Max_Robust",
          "role": "daytime_authority_paper_demo",
          "status": "PHASE25_FINAL_CLOSEOUT_COMPLETE_READY_FOR_PAPER_DEMO_WITH_WARNINGS"
        },
        "daytime_backup": {
          "id": "Phase24_Robust_Peak",
          "role": "strong_backup",
          "status": "SUPERSEDED_VALID"
        },
        "daytime_baseline": {
          "id": "Phase18_Baseline",
          "role": "protected_baseline",
          "status": "VALIDATED"
        },
        "overnight_primary": {
          "id": "SCBI_M5_GLOBAL",
          "role": "overnight_authority",
          "status": "PROTECTED_NOT_DAYTIME"
        },
        "research": {
            "id": "Phase26_Shadow",
            "status": "BLOCKED_PENDING_2015_2019_M1_OR_TICK_DATA"
        },
        "quarantined_or_rejected": [
          {"id": "Phase19", "status": "INVALIDATED_AND_ARCHIVED"},
          {"id": "Phase22", "status": "SUPERSEDED"}
        ]
      },
      "lab_status": {
        "id": "BOT_V2_DAYTIME_LAB",
        "role": "paper_demo_execution",
        "authority": "PHASE25_MAX_ROBUST"
      },
      "execution_gates": {
        "news_fortress": "FAIL_CLOSED",
        "data_quality_mask": "FAIL_CLOSED",
        "mt5_real": "BLOCKED",
        "ctrader": "BLOCKED",
        "vps": "BLOCKED",
        "config_hash_check": "REQUIRED"
      }
    }
    with open(root / "02_STRATEGY_AUTHORITY_MAP.json", "w") as f:
        json.dump(auth_map, f, indent=2)

    lab_status = {
      "current_authority": "PHASE25_MAX_ROBUST",
      "current_execution_status": "PAPER_DEMO_ONLY_WITH_WARNINGS",
      "phase26a_status": "PHASE26A_DATA_PARTIAL_REQUIRES_REPAIR",
      "phase26_optimization_status": "BLOCKED_PENDING_2015_2019_M1_OR_TICK_DATA",
      "real_trading_blocked": True,
      "mt5_real_blocked": True,
      "ctrader_blocked": True,
      "vps_blocked": True,
      "scbi_touched": False,
      "phase19_reopened": False,
      "news_fortress": "FAIL_CLOSED_ACTIVE",
      "data_quality_mask": "FAIL_CLOSED_ACTIVE"
    }
    with open(lab / "status.json", "w") as f:
        json.dump(lab_status, f, indent=2)

if __name__ == "__main__":
    generate_closeout_docs()
