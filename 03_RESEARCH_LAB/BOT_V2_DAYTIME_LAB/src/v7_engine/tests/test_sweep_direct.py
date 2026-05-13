import pytest
from pathlib import Path
from src.v7_engine.sweep_direct import SweepDirectOrchestrator

def test_sweep_direct_generates_all_mandatory_outputs(tmp_path):
    """
    Verifica: test_sweep_direct_generates_all_mandatory_outputs.
    El orquestador maestro procesa la matriz hiper-dimensional y emite
    estrictamente la totalidad de los 15 archivos exigidos por la Sección 11.
    """
    # Usar un subdirectorio aislado temporal para la prueba
    test_out = tmp_path / "reports/v37_manipulante2"
    orchestrator = SweepDirectOrchestrator(output_dir=test_out)
    
    # Ejecutar la orquestación y emisión de archivos
    orchestrator.execute_deterministic_sweep()
    
    # Listado oficial inalterable de la Sección 11
    mandatory_files = [
        "V37_SEARCH_SPACE.json",
        "V37_TRAIN_RESULTS.csv",
        "V37_TRAIN_TOP5.json",
        "V37_VAL_RESULTS.csv",
        "V37_VAL_WINNER.json",
        "V37_TEST_TRADES.csv",
        "V37_TEST_METRICS.json",
        "V37_MAE_AUDIT.json",
        "V37_D6_SELFCHECK.json",
        "V37_FTMO_COMPLIANCE.json",
        "V37_NEWS_LOG.json",
        "V37_INDEPENDENT_VERIFY.json",
        "V37_FINAL_VERDICT.md",
        "V37_PYTEST_OUTPUT.txt",
        "V37_RUNTIME_LOG.txt"
    ]
    
    # Certificar existencia unívoca
    for filename in mandatory_files:
        target_file = orchestrator.output_dir / filename
        assert target_file.exists(), f"Fallo contractual: Archivo obligatorio faltante {filename}"
        assert target_file.stat().st_size > 0, f"Fallo de contenido: Archivo vacío {filename}"
