import pytest
import pandas as pd
import numpy as np
from src.R1.r1_detector import R1AbsorptionDetector
from src.R1.r1_levels import R1LevelExtractor

def test_extract_r1_levels_structure():
    """Verifica que el extractor de niveles devuelva la estructura de datos esperada para R1."""
    # Crear un DataFrame OHLCV sintético causal
    dates = pd.date_range("2026-05-01 07:00:00", periods=10, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "open": np.linspace(1.1000, 1.1050, 10),
        "high": np.linspace(1.1010, 1.1060, 10),
        "low": np.linspace(1.0990, 1.1040, 10),
        "close": np.linspace(1.1005, 1.1055, 10),
        "volume": np.random.randint(100, 1000, 10)
    }, index=dates)
    
    extractor = R1LevelExtractor()
    levels = extractor.get_levels(df)
    assert isinstance(levels, pd.DataFrame), "El resultado debe ser un DataFrame indexado."
    assert "pdh" in levels.columns, "Debe contener el nivel máximo del día previo (PDH)."
    assert "pdl" in levels.columns, "Debe contener el nivel mínimo del día previo (PDL)."

def test_r1_absorption_detector_initialization():
    """Verifica la inicialización causal y el blindaje paramétrico del detector de absorción."""
    detector = R1AbsorptionDetector(wick_to_body_min=2.5, close_back_inside=True)
    assert detector.wick_to_body_min == 2.5, "El umbral de mecha a cuerpo debe asignarse nativamente."
    assert detector.close_back_inside is True, "El flag de cierre interno debe conservarse."
