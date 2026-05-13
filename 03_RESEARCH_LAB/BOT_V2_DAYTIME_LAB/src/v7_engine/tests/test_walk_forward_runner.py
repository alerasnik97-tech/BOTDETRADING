import pytest
import json
from pathlib import Path
from src.v7_engine.walk_forward_runner import WalkForwardRunner

def test_walk_forward_runner_processes_train_and_top5(tmp_path):
    """
    Verifica: test_walk_forward_runner_processes_train_and_top5.
    El runner inmutable ejecuta el entrenamiento y extrae correctamente
    el Top 5 asegurando que superen el umbral contractual PF >= 1.5.
    """
    test_rep = tmp_path / "reports/v37_manipulante2"
    runner = WalkForwardRunner(reports_dir=test_rep)
    
    df_train = runner.evaluate_train_partition()
    
    # Verificar persistencia de V37_TRAIN_RESULTS.csv
    csv_file = runner.reports_dir / "V37_TRAIN_RESULTS.csv"
    assert csv_file.exists(), "Fallo de salida: V37_TRAIN_RESULTS.csv no generado"
    
    # Verificar Top 5 JSON
    top5_file = runner.reports_dir / "V37_TRAIN_TOP5.json"
    assert top5_file.exists(), "Fallo de salida: V37_TRAIN_TOP5.json no generado"
    
    with open(top5_file) as f:
        top5 = json.load(f)
        
    assert len(top5) <= 5, "Fallo contractual: Exceso de candidatas en el Top 5"
    for cand in top5:
        assert cand["PF"] >= 1.50, f"Fallo de potencia: Candidata {cand['id']} con PF < 1.5"

def test_walk_forward_runner_processes_validation_winner(tmp_path):
    """
    Verifica: test_walk_forward_runner_processes_validation_winner.
    Se somete a las candidatas al período de validación y se extrae
    la única ganadora certificada para la fase ciega.
    """
    test_rep = tmp_path / "reports/v37_manipulante2"
    runner = WalkForwardRunner(reports_dir=test_rep)
    
    winner = runner.evaluate_validation_partition()
    
    # Verificar salidas
    val_csv = runner.reports_dir / "V37_VAL_RESULTS.csv"
    assert val_csv.exists(), "Fallo de salida: V37_VAL_RESULTS.csv no generado"
    
    val_win = runner.reports_dir / "V37_VAL_WINNER.json"
    assert val_win.exists(), "Fallo de salida: V37_VAL_WINNER.json no generado"
    
    assert "id" in winner, "Fallo de estructura: Ganadora carece de identificador unívoco"
    assert winner["PF_val"] >= 1.30, "Fallo de paso: Ganadora de validación con PF_val inferior a 1.3"
