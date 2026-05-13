# CERTIFICACIÓN DE PREFLIGHT LIMPIO R1 POST-RESTAURACIÓN DEL MOTOR

## 1. Identificadores de Estado Inmutable
- **Runner**: `run_r1_micro_probe.py` (Adaptado para smoke test de 1 mes: `2020-01`).
- **Engine Core**: `src/v7_engine/engine.py` (Restaurado canónicamente desde la rama estable institucional `agent/research-manipulante4-sweep-quality`).
- **Cost Model**: `src/v7_engine/cost_model.py` (Paridad 100% canónica restaurada).
- **V6 Utils**: `src/v6_utils/` (Módulos causales inmutables de construcción de barras y manejo de memoria restaurados).

## 2. Parámetros y Restricciones Verificadas durante Ejecución
- **Inmutabilidad de Fuentes**: Cero ediciones de código en directorios del motor durante el preflight.
- **Throttler Institucional**: Límite de `max_trades_per_day = 3` operando de forma activa y bloqueando sobre-operación.
- **Ventana de Sesión**: Ingresos acotados estrictamente entre las `07:00` y las `11:00` NY, con cierre de forzado diario activado para las `16:55` NY.
- **Filtro de Noticias (News Calendar)**: Carga exitosa de eventos de impacto alto/medio desde `news_eurusd_am_fortress_v3.csv` aplicando bloqueos en buffers pre/post evento.
- **Cierre Artificial a Fin de Mes (EOM)**: Las métricas `net_r` y comisiones son calculadas de forma rigurosa por el modelo de costos institucional sin incluir truncamientos irreales en el desempeño de la estrategia.

## 3. Evidencia Física
- **Trades Generados**: `R1_MICRO_PROBE_TRADES.csv` (300.8 KB, representando las ejecuciones válidas del mes de Enero 2020 a lo largo de las 54 configuraciones).
- **Estado**: Preflight concluido sin fallas de memoria, sin advertencias de look-ahead y cumpliendo el protocolo `exit code: 0`.
