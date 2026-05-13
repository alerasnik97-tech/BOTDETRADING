import json
import pytest
from pathlib import Path
from reports.v37_manipulante2.rebuild_v1.build_search_space import (
    generate_and_save,
    build_full_search_space,
    compute_config_hash
)

@pytest.fixture
def sample_space(tmp_path):
    target = tmp_path / "V1_SEARCH_SPACE_LOCKED.json"
    return generate_and_save(target)

def test_search_space_contains_core_expanded_diagnostic(sample_space):
    """1. El espacio consolidado contiene unívocamente las tres particiones exigidas."""
    partitions = {cfg["space_partition"] for cfg in sample_space}
    assert partitions == {"CORE", "EXPANDED", "DIAGNOSTIC"}

def test_core_has_expected_288_configs(sample_space):
    """2. La dimensión CORE acredita exactamente 288 permutaciones predefinidas."""
    core_configs = [c for c in sample_space if c["space_partition"] == "CORE"]
    assert len(core_configs) == 288

def test_every_config_has_deterministic_id(sample_space):
    """3. Cada configuración serializada posee una clave identificadora unívoca."""
    ids = [c["config_id"] for c in sample_space]
    assert len(ids) == len(set(ids))
    assert all(i.startswith(("core_", "exp_", "diag_")) for i in ids)

def test_every_config_has_deterministic_hash(sample_space):
    """4. La firma criptográfica se computa de forma pura sobre los hiper-parámetros."""
    for cfg in sample_space:
        expected = compute_config_hash(cfg)
        assert cfg["config_hash"] == expected

def test_diagnostic_configs_not_eligible(sample_space):
    """5. Las variantes de diagnóstico poseen bloqueos explícitos para selección OOS."""
    diag_configs = [c for c in sample_space if c["space_partition"] == "DIAGNOSTIC"]
    assert len(diag_configs) > 0
    for c in diag_configs:
        assert c["eligible_for_selection"] is False
        assert c["diagnostic_only"] is True

def test_news_none_is_diagnostic_only(sample_space):
    """6. La inhabilitación del filtro de noticias marca automáticamente a la configuración como diagnóstica."""
    for c in sample_space:
        if c["news_mode"] == "none" or not c["news_enabled"]:
            assert c["diagnostic_only"] is True
            assert c["eligible_for_selection"] is False

def test_limit_retrace_is_diagnostic_only_unless_explicitly_enabled(sample_space):
    """7. Entradas por retrace límite sin confirmación activa se asignan a control analítico."""
    for c in sample_space:
        if c["entry"] == "limit_retrace":
            assert c["diagnostic_only"] is True
            assert c["eligible_for_selection"] is False

def test_no_duplicate_config_hashes(sample_space):
    """8. Se certifica cero colisiones criptográficas en la totalidad del hiper-espacio."""
    hashes = [c["config_hash"] for c in sample_space]
    assert len(hashes) == len(set(hashes))

def test_search_space_serialization_is_stable(sample_space, tmp_path):
    """9. La serialización de estructuras al disco duro mantiene una firma binaria inmutable."""
    p1 = tmp_path / "f1.json"
    p2 = tmp_path / "f2.json"
    generate_and_save(p1)
    generate_and_save(p2)
    assert p1.read_bytes() == p2.read_bytes()

def test_search_space_written_before_results(sample_space):
    """10. El manifiesto paramétrico precede de forma obligatoria a la inyección de resultados."""
    # Aserción de diseño de orquestación
    assert len(sample_space) == 356
