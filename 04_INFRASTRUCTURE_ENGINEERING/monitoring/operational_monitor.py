"""
SISTEMA DE INTELIGENCIA OPERATIVA - MONITOREO AUTOMATICO
Fase 2: Capa de control externa sobre laboratorio en produccion
Solo lectura del lab, escritura externa de logs
"""
import json
import csv
import sys
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional

# RUTAS
LAB_ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
MONITOR_ROOT = Path(r"C:\Users\alera\Desktop\Bot\operational_logs")
LOG_FILE = MONITOR_ROOT / "daily_operational_log.csv"

# Fuentes de datos del laboratorio
STATUS_SOURCES = {
    "pipeline": LAB_ROOT / "DATA_COVERAGE_PIPELINE_STATUS.json",
    "chain": LAB_ROOT / "DAILY_DATA_TO_DECISION_CHAIN_STATUS.json",
    "autopilot": LAB_ROOT / "SCBI_PHASE1_AUTOPILOT_STATUS.json",
}


@dataclass
class DailyOperationalRecord:
    date: str
    feeder_status: str
    autopilot_status: str
    promotion_status: str
    chain_status: str
    classification: str
    coverage_ready: bool
    daily_operable: bool
    blockers: str
    bundle_updated: bool
    

def classify_day(pipeline_data: dict, chain_data: dict) -> str:
    """
    Clasificacion inteligente basada en datos reales del sistema.
    No hardcode - deriva de condiciones reales.
    """
    # Extraer estados
    pipeline_decision = pipeline_data.get("decision", "UNKNOWN")
    pipeline_taxonomy = pipeline_data.get("taxonomy_outcome", "UNKNOWN")
    chain_verdict = chain_data.get("final_verdict", "UNKNOWN")
    coverage_blockers = pipeline_data.get("coverage_blockers", [])
    coverage_ready = pipeline_data.get("coverage_ready", False)
    daily_operable = pipeline_data.get("daily_operable", False)
    
    # Check promotion results si existen
    promotion_results = pipeline_data.get("promotion_results", [])
    has_staging = any(r.get("status") == "STAGING" for r in promotion_results)
    has_promoted = any(r.get("status") == "PROMOTED" for r in promotion_results)
    has_block = any(r.get("status") == "BLOCK" for r in promotion_results)
    
    # Logica de clasificacion
    
    # 1. Error real de automatizacion
    if has_block and any("SCHEMA" in str(b) or "INVALID" in str(b) for b in coverage_blockers):
        return "AUTOMATION_BLOCKED_BY_REAL_ERROR"
    
    # 2. Ejecucion completa del chain
    if chain_verdict == "SUCCESS" and daily_operable:
        return "DAILY_CHAIN_EXECUTED"
    
    # 3. Cobertura restaurada (nuevos datos promovidos)
    if has_promoted and coverage_ready:
        return "COVERAGE_RESTORED"
    
    # 4. Data refresh canonical (overlap detectado, no duplicacion)
    if has_staging and "OVERLAPS_CANONICAL" in str(promotion_results):
        return "DATA_REFRESH_CANONICAL"
    
    # 5. Baseline pass ready for chain
    if coverage_ready and pipeline_decision == "PASS" and chain_verdict != "SUCCESS":
        return "BASELINE_PASS_READY_FOR_CHAIN"
    
    # 6. Bloqueo correcto del sistema
    if pipeline_decision == "BLOCK" and coverage_blockers:
        # Verificar si es proteccion legitima o error
        if all("NO_TARGET_ROWS" in str(b) or "INSUFFICIENT" in str(b) for b in coverage_blockers):
            return "FAIL_CLOSED_CORRECT_BEHAVIOR"
    
    # Caso no clasificado - revisar manualmente
    return f"UNCLASSIFIED_{pipeline_taxonomy}_{chain_verdict}"


def extract_blockers(pipeline_data: dict) -> str:
    """Extrae bloqueadores como string separado por pipe."""
    blockers = pipeline_data.get("coverage_blockers", [])
    validator_blockers = pipeline_data.get("validator_integration", {}).get("blockers", [])
    all_blockers = blockers + validator_blockers
    return "|".join(all_blockers) if all_blockers else "NONE"


def extract_promotion_status(pipeline_data: dict) -> str:
    """Resume el estado de promotion de todos los datasets."""
    results = pipeline_data.get("promotion_results", [])
    if not results:
        return "NO_DATA"
    
    statuses = [f"{r.get('kind')}:{r.get('status')}" for r in results]
    return "|".join(statuses)


def parse_autopilot_status() -> str:
    """Lee el ultimo estado conocido del autopilot."""
    autopilot_file = STATUS_SOURCES["autopilot"]
    if not autopilot_file.exists():
        return "FILE_MISSING"
    
    try:
        data = json.loads(autopilot_file.read_text(encoding="utf-8"))
        last_run = data.get("last_run", {})
        return last_run.get("verdict", "UNKNOWN")
    except Exception:
        return "PARSE_ERROR"


def parse_chain_status() -> str:
    """Lee el estado de la cadena."""
    chain_file = STATUS_SOURCES["chain"]
    if not chain_file.exists():
        return "FILE_MISSING"
    
    try:
        data = json.loads(chain_file.read_text(encoding="utf-8"))
        return data.get("final_verdict", "UNKNOWN")
    except Exception:
        return "PARSE_ERROR"


def check_bundle_updated() -> bool:
    """Verifica si el bundle se actualizo recientemente (ultimas 24h)."""
    bundle_file = LAB_ROOT / "000_PARA_CHATGPT.zip"
    if not bundle_file.exists():
        return False
    
    mtime = datetime.fromtimestamp(bundle_file.stat().st_mtime, tz=timezone.utc)
    now = datetime.now(timezone.utc)
    hours_since_update = (now - mtime).total_seconds() / 3600
    
    return hours_since_update < 24


def capture_daily_record(target_date: Optional[str] = None) -> DailyOperationalRecord:
    """
    Captura el estado operativo del dia.
    Lee SOLO archivos del laboratorio, no modifica nada.
    """
    # Determinar fecha objetivo
    if target_date is None:
        # Usar fecha del pipeline si existe, o ayer
        pipeline_file = STATUS_SOURCES["pipeline"]
        if pipeline_file.exists():
            try:
                data = json.loads(pipeline_file.read_text(encoding="utf-8"))
                target_date = data.get("target_date", "")
            except Exception:
                target_date = ""
        if not target_date:
            from datetime import timedelta
            target_date = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    
    # Leer datos del pipeline
    pipeline_data = {}
    pipeline_file = STATUS_SOURCES["pipeline"]
    if pipeline_file.exists():
        try:
            pipeline_data = json.loads(pipeline_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Error leyendo pipeline status: {e}")
    
    # Leer datos de chain
    chain_data = {}
    chain_file = STATUS_SOURCES["chain"]
    if chain_file.exists():
        try:
            chain_data = json.loads(chain_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Error leyendo chain status: {e}")
    
    # Verificar feeder (intake files)
    intake_path = LAB_ROOT / "data" / "coverage_pipeline" / "intake"
    feeder_ok = any(intake_path.glob("*.csv")) if intake_path.exists() else False
    feeder_status = "PRESENT" if feeder_ok else "EMPTY"
    
    # Clasificar el dia
    classification = classify_day(pipeline_data, chain_data)
    
    # Extraer estados
    promotion_status = extract_promotion_status(pipeline_data)
    autopilot_status = parse_autopilot_status()
    chain_status = parse_chain_status()
    blockers = extract_blockers(pipeline_data)
    coverage_ready = pipeline_data.get("coverage_ready", False)
    daily_operable = pipeline_data.get("daily_operable", False)
    bundle_updated = check_bundle_updated()
    
    return DailyOperationalRecord(
        date=target_date,
        feeder_status=feeder_status,
        autopilot_status=autopilot_status,
        promotion_status=promotion_status,
        chain_status=chain_status,
        classification=classification,
        coverage_ready=coverage_ready,
        daily_operable=daily_operable,
        blockers=blockers,
        bundle_updated=bundle_updated,
    )


def write_record_to_csv(record: DailyOperationalRecord):
    """Escribe el registro al CSV. Crea archivo si no existe."""
    MONITOR_ROOT.mkdir(parents=True, exist_ok=True)
    
    file_exists = LOG_FILE.exists()
    
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "date", "feeder_status", "autopilot_status", "promotion_status",
            "chain_status", "classification", "coverage_ready", "daily_operable",
            "blockers", "bundle_updated"
        ])
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(asdict(record))


def generate_metrics():
    """Genera metricas acumuladas desde el log."""
    if not LOG_FILE.exists():
        print("[INFO] No hay log historico todavia.")
        return
    
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)
    
    if not records:
        print("[INFO] Log vacio.")
        return
    
    total_days = len(records)
    
    # Conteos por clasificacion
    from collections import Counter
    classifications = Counter(r["classification"] for r in records)
    coverage_ready_days = sum(1 for r in records if r["coverage_ready"] == "True")
    daily_operable_days = sum(1 for r in records if r["daily_operable"] == "True")
    bundle_updated_days = sum(1 for r in records if r["bundle_updated"] == "True")
    
    print("\n=== METRICAS OPERATIVAS ACUMULADAS ===")
    print(f"Total dias registrados: {total_days}")
    print(f"\nDistribucion por clasificacion:")
    for cls, count in classifications.most_common():
        pct = (count / total_days) * 100
        print(f"  {cls}: {count} ({pct:.1f}%)")
    
    print(f"\nIndicadores:")
    print(f"  Coverage ready: {coverage_ready_days}/{total_days} ({(coverage_ready_days/total_days)*100:.1f}%)")
    print(f"  Daily operable: {daily_operable_days}/{total_days} ({(daily_operable_days/total_days)*100:.1f}%)")
    print(f"  Bundle actualizado: {bundle_updated_days}/{total_days} ({(bundle_updated_days/total_days)*100:.1f}%)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Monitoreo Operativo del Laboratorio")
    parser.add_argument("--date", help="Fecha objetivo YYYY-MM-DD (default: auto-detect)")
    parser.add_argument("--metrics", action="store_true", help="Mostrar metricas acumuladas")
    args = parser.parse_args()
    
    if args.metrics:
        generate_metrics()
        return
    
    print("=== CAPTURA OPERATIVA DIARIA ===")
    print(f"[INFO] Leyendo estado del laboratorio...")
    
    record = capture_daily_record(args.date)
    
    print(f"\n[CAPTURE] {record.date}")
    print(f"  Feeder: {record.feeder_status}")
    print(f"  Autopilot: {record.autopilot_status}")
    print(f"  Promotion: {record.promotion_status}")
    print(f"  Chain: {record.chain_status}")
    print(f"  Classification: {record.classification}")
    print(f"  Coverage ready: {record.coverage_ready}")
    print(f"  Daily operable: {record.daily_operable}")
    print(f"  Bundle updated: {record.bundle_updated}")
    
    write_record_to_csv(record)
    print(f"\n[OK] Registro guardado en: {LOG_FILE}")


if __name__ == "__main__":
    main()
