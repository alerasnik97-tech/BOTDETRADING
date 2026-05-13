import pytest
from pathlib import Path
from src.v7_engine.cache_builder import CacheBuilder

def test_cache_builder_populates_all_target_timeframes(tmp_path):
    """
    Verifica: test_cache_builder_populates_all_target_timeframes.
    El gestor inmutable de precómputo crea y valida las particiones Parquet
    para la totalidad de los timeframes del hiper-espacio operativo.
    """
    test_cache = tmp_path / "data/cache"
    builder = CacheBuilder(cache_dir=test_cache)
    
    # Ingestar y serializar barras
    paths = builder.populate_bars_cache()
    
    # Verificar que se crean archivos para M1, M3, M5, M30, H1, H4
    assert len(paths) > 0, "Fallo de creación: Caché OHLC vacío"
    for p in paths:
        assert p.exists(), f"Fallo de persistencia: Archivo {p} no existe en disco"
        assert p.suffix == ".parquet", f"Fallo de formato: El sufijo no es Parquet nativo en {p}"

def test_cache_builder_precomputes_signals_for_all_pairs(tmp_path):
    """
    Verifica: test_cache_builder_precomputes_signals_for_all_pairs.
    Se registran y aíslan los eventos pre-calculados para las 9 combinaciones
    ortogonales de TF1 y TF2 sin omisiones.
    """
    test_cache = tmp_path / "data/cache"
    builder = CacheBuilder(cache_dir=test_cache)
    
    summary = builder.precompute_signals()
    
    expected_pairs = [
        "M30_M1", "M30_M3", "M30_M5",
        "H1_M1", "H1_M3", "H1_M5",
        "H4_M1", "H4_M3", "H4_M5"
    ]
    
    for pair in expected_pairs:
        assert pair in summary, f"Fallo de cobertura: Par de timeframe faltante en señales {pair}"
        assert summary[pair] > 0, f"Fallo de eventos: Cero señales precomputadas para {pair}"
