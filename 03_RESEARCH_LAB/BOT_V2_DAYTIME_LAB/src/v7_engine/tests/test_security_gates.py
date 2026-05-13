import pytest
from pathlib import Path
from src.v7_engine.security_gates import (
    guard_file_write,
    guard_ticks_readonly,
    NetworkGuard,
    guard_modify_core_file,
    guard_git_phase_ready,
    SecurityWhitelistError,
    SecurityReadOnlyError,
    SecurityNetworkBlockedError,
    SecurityBackupMissingError,
    SecurityGitCommitError
)

def test_no_write_outside_whitelist():
    """
    Verifica: Intento de escribir en /Desktop/ o C:\ raíz lanza excepción.
    Propiedad de Aislamiento de Archivos (Sección 2.1).
    """
    with pytest.raises(SecurityWhitelistError) as exc_info:
        guard_file_write("C:/Desktop/malicious_output.txt")
    assert "fuera de whitelist" in str(exc_info.value).lower()
    
    with pytest.raises(SecurityWhitelistError):
        guard_file_write("C:/system_root_file.sys")
        
    # Rutas permitidas no deben lanzar excepción
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    valid_path = base_dir / "reports/v37_manipulante2/test_output.csv"
    # Llamado seguro
    try:
        guard_file_write(valid_path, base_lab_dir=base_dir)
    except SecurityWhitelistError:
        pytest.fail("Ruta autorizada en whitelist lanzó excepción inesperadamente.")

def test_ticks_dir_readonly():
    """
    Verifica: Intento de modificar archivo en data/dukascopy/ falla.
    Propiedad de Integridad de Datos Externos (Sección 2.1).
    """
    tick_file = "data/dukascopy/EURUSD_2026_01.parquet"
    
    # Lectura permitida
    try:
        guard_ticks_readonly(tick_file, mode="rb")
        guard_ticks_readonly(tick_file, mode="r")
    except SecurityReadOnlyError:
        pytest.fail("El modo de lectura pura lanzó excepción erróneamente.")
        
    # Escritura / Modificación denegada
    with pytest.raises(SecurityReadOnlyError) as exc_info:
        guard_ticks_readonly(tick_file, mode="wb")
    assert "solo lectura" in str(exc_info.value).lower()
    
    with pytest.raises(SecurityReadOnlyError):
        guard_ticks_readonly(tick_file, mode="a")

def test_network_blocked_except_calendar():
    """
    Verifica: Request a google.com falla, request a nfs.faireconomy.media pasa.
    Propiedad de Aislamiento de Red (Sección 2.2).
    """
    guard = NetworkGuard()
    
    with pytest.raises(SecurityNetworkBlockedError) as exc_info:
        guard.safe_request("https://www.google.com/search?q=forex")
    assert "google.com" in str(exc_info.value).lower() or "denegado" in str(exc_info.value).lower()
    
    # Excepción explícitamente permitida
    response = guard.safe_request("https://nfs.faireconomy.media/ff_calendar_thisweek.json")
    assert response == "MOCK_RESPONSE_OK"

def test_backup_before_modify(tmp_path):
    """
    Verifica: Modificar v6_utils/execution.py sin backup previo lanza excepción.
    Propiedad de Defensa en Profundidad (Sección 2.3).
    """
    core_file = "src/v6_utils/execution.py"
    backups_dir = tmp_path / "backups"
    backups_dir.mkdir()
    
    # Caso 1: Directorio de backups vacío
    with pytest.raises(SecurityBackupMissingError) as exc_info:
        guard_modify_core_file(core_file, backups_root=backups_dir)
    assert "resguardo previo" in str(exc_info.value).lower()
    
    # Caso 2: Resguardo válido materializado
    valid_backup = backups_dir / "v6_utils_20260512_122200"
    valid_backup.mkdir()
    
    try:
        guard_modify_core_file(core_file, backups_root=backups_dir)
    except SecurityBackupMissingError:
        pytest.fail("Fallo al reconocer un backup válido de v6_utils existente.")

def test_git_commit_before_phase(monkeypatch):
    """
    Verifica: Empezar una fase sin commit limpio lanza excepción.
    Propiedad de Trazabilidad Obligatoria (Sección 2.1).
    """
    # Para aislar el test y que sea reproducible y robusto en CI/local,
    # simulamos la salida de git log
    import subprocess
    
    def mock_run(*args, **kwargs):
        class CompletedProcessMock:
            stdout = "5ec0cb0 [v37/fase0] pre-compromiso manipulante 2.0\n"
            returncode = 0
        return CompletedProcessMock()
        
    monkeypatch.setattr(subprocess, "run", mock_run)
    
    try:
        guard_git_phase_ready(Path("."))
    except SecurityGitCommitError:
        pytest.fail("Se rechazó un commit limpio de la fase v37 válido.")
        
    # Caso de fallo: commit ajeno o faltante
    def mock_run_fail(*args, **kwargs):
        class CompletedProcessMockFail:
            stdout = "a1b2c3d fix typos in readme\n"
            returncode = 0
        return CompletedProcessMockFail()
        
    monkeypatch.setattr(subprocess, "run", mock_run_fail)
    
    with pytest.raises(SecurityGitCommitError) as exc_info:
        guard_git_phase_ready(Path("."))
    assert "commit obligatorio" in str(exc_info.value).lower()
