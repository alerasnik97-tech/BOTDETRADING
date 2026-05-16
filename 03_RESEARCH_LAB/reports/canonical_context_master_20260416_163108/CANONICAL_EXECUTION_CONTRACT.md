# CONTRATO CANÓNICO DE EJECUCIÓN (Canonical Execution Contract)

## 1. RUNNER OFICIAL
El **único runner oficial** habilitado para corridas serias de discovery, research o validación paramétrica final es:
`research_lab/main.py`
* **Runners NO oficiales:** `light_runner.py` (Solo para depuración de dependencias), blocks interactivos de Jupyter, o scripts `_temp_*` confinados en el Legacy Archive.

## 2. PARÁMETROS Y FLAGS OFICIALES
Toda ejecución formal debe invocarse vía CLI (o wrapper equivalente) asegurando:
- `--strategy [nombre]`
- `--end 2025-12-31` (Para abarcar toda la muestra viva de datos OOS).
* Nota: El Rejection Harness corre incrustado **por defecto** en `main.py`. No requiere flag adicional y NUNCA debe ser puenteado por código manual.

## 3. OUTPUTS Y REPORTES OFICIALES
Los resultados válidos únicamente están autorizados a persistir en:
`results/research_lab_robust/[timestamp]_[strategy_name]/`

### Archivos canónicos generados obligatoriamente:
- **`lineage_metadata.json`**: Guarda el rastro inmutable (runner, strategy, params base, is/oos lengths, environment y flags temporales). Es la prueba de paternidad del backtest.
- **`REJECTION_REPORT.md`**: (Sí la estrategia fue aniquilada). Detalla qué umbral mató a la estrategia matemáticamente (`IN_SAMPLE_FLOP`, `OOS_WFA_FLOP`, o `OOS_DRAWDOWN`).
- **`PARA CHATGPT/`**: Conjunto de gráficas (Equity, Drawdown, Heatmaps) reservado sólo si superaron el escrutinio Base.
- **`rejection_summary_log.csv`** (En root de results): Si se invoca `run-all`, tabula el dictamen absoluto de la granja evaluada.

## 4. METADATA ESTRICTA Y LINAJE
Las decisiones no se toman sobre reportes ciegos. Todo output canónico incluirá obligatoriamente un JSON reflejando:
1. Hora de ejecución y nombre del framework F1 base.
2. Modos de motor activo en el instante (`high_precision_mode` vs `normal_mode`).
3. Multiplicadores de costo de sesión asimilados.
Si un output en la carpeta de reportes no contiene un `lineage_metadata.json` válido, **NO SE CONSIDERA MATERIAL PROBATORIO** y debe asumirse obsoleto/adulterado.
