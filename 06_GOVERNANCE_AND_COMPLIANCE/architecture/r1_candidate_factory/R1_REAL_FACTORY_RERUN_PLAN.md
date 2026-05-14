# PLAN DE RE-EJECUCIÓN REAL DE CANDIDATE FACTORY — R1

## 1. Requisitos de Autenticidad (Anti-Placeholder)
La nueva ejecución de la Candidate Factory debe producir evidencia física incontrovertible:
- **Ranking Real**: Archivo `R1_REAL_FACTORY_RANKING.csv` con 1200 configuraciones reales y métricas calculadas.
- **Trazabilidad**: Inclusión de un `RUN_LOG.txt` con marcas de tiempo de cada backtest.
- **Transaccionalidad**: El archivo de trades debe contener la totalidad de las operaciones liquidadas para las top configuraciones.

## 2. Metodología de Validación
No se aceptará ningún resultado que no pase el filtro de **Independent Verify** recalculado directamente desde el archivo de trades completo.

## 3. Blindaje de Motor
Se mantiene la exigencia de paridad `ENGINE_CORE_OK` antes y después de la corrida.
