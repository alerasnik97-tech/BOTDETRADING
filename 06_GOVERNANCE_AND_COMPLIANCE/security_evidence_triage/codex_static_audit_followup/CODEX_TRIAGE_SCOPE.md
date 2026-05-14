# Codex Triage Scope - V50.1

## Alcance de la Auditoría
1. **Secret Triage**: Escaneo de tokens de Telegram, GitHub, API keys y credenciales en archivos `.env`, `.py`, `.json`, `.md` y reportes históricos.
2. **Evidence Integrity**: Identificación de contradicciones entre reportes legacy (ZIP-based) y el estado actual (GitHub-based).
3. **Mock Surface**: Auditoría de módulos de "evidencia sintética" (`sweep_direct.py`, `walk_forward_runner.py`).
4. **Data/News Manifest**: Verificación de cobertura de manifiestos y hashes en `05_MARKET_DATA_VAULT`.
5. **Static Code Risks**:
    - Timezone conversion risks in `temporal.py` / `data_loader.py`.
    - FTMO Floating Drawdown monitoring gaps in `engine.py`.
    - Slippage realism in `cost_model.py`.

## Exclusiones
- No se realiza corrección de código.
- No se realiza saneamiento de Git.
- No se ejecutan pruebas dinámicas.
