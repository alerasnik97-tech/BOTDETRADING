import os
import sys
from datetime import datetime
from shadow_line_lab import config, runner_shadow, ledger_io, telemetry, reporting

class ShadowOrchestrator:
    def __init__(self):
        self.runner = runner_shadow.ShadowRunner(config.STRATEGY_CONFIG)
        telemetry.log_event("init", "Shadow Orchestrator Initialized")

    def run_daily_process(self, date_str, h1_data=None, m5_data=None, news_data=None, levels=None):
        """
        Punto de entrada para la ejecución diaria de la Shadow Line.
        """
        telemetry.log_event("run_start", f"Processing date: {date_str}")
        
        # En una implementación real, aquí cargaríamos los datos de config.PRICE_DIRS
        # Para este nivel de infraestructura, el orquestador espera recibir los datos
        # o fallar con gracia si es una ejecución de prueba.
        
        if h1_data is None:
            telemetry.log_event("warning", "No H1 data provided. Dry run mode.")
            results = {"date": date_str, "classification": "DRY_RUN_NO_DATA", "pnl_r": 0.0}
        else:
            results = self.runner.run_daily_check(date_str, h1_data, m5_data, news_data, levels)
        
        # Guardar resultados
        ledger_io.write_ledger_entry(results, config.LEDGER_FILE)
        
        status = {
            "last_run": datetime.utcnow().isoformat() + "Z",
            "last_date_processed": date_str,
            "last_result": results["classification"]
        }
        ledger_io.save_json(status, config.DAILY_STATUS_FILE)
        
        telemetry.log_event("run_end", f"Finished date: {date_str} with result: {results['classification']}")
        
        # Generar reporte sumario
        summary = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "trades_executed": 1 if results["classification"] == "TRADE_EXECUTED" else 0,
            "total_pnl_r": results.get("pnl_r", 0.0)
        }
        ledger_io.save_json(summary, config.SUMMARY_FILE)
        reporting.generate_shadow_report(summary)
        
        return results

if __name__ == "__main__":
    orchestrator = ShadowOrchestrator()
    today = datetime.utcnow().strftime("%Y-%m-%d")
    print(f"Iniciando Shadow Line para {today}...")
    orchestrator.run_daily_process(today)
    print("Shadow Line ejecutada. Ver resultados en shadow_line_lab/results/")
