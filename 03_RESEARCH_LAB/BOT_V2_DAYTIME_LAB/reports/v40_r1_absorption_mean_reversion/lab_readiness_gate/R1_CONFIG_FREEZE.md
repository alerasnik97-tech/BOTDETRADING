# CONGELAMIENTO DEL ESPACIO PARAMÉTRICO (CONFIG FREEZE)

## 1. Parámetros e Instrumento
- **Instrumento Único Autorizado**: `EURUSD` (Cero optimización cruzada o selección de activos a posteriori).
- **Conteo de Configuraciones**: Total de `54` configuraciones activas (Mapeadas en `R1_MICRO_PROBE_RUN_CONFIG.json`), respetando estrictamente el límite institucional de `configs <= 150`.

## 2. Horarios y Frecuencia Operativa
- **Ventana Operativa Máxima**: Ingresos restringidos de `07:00` a `17:00` NY.
- **Ventana de Alta Prioridad**: Foco algorítmico principal concentrado en la sesión de NY Open de `08:00` a `11:00` NY.
- **Throttler Institucional**: Límite estricto de `max_trades_per_day = 3` implementado y operando de forma activa en la capa superior del orquestador.

## 3. Costos y Filtros Inmutables
- **Slippage Obligatorio**: Costo de deslizamiento fijo de `0.2` pips impuesto de forma ineludible en el cálculo de rendimiento neto.
- **Comisiones FTMO**: Estructura de comisiones redondas ($5.0 por lote estándar) aplicada nativamente.
- **Filtro de Noticias (Data/News)**: Buffers de exclusión temporal activos sobre el calendario macroeconómico institucional.
- **Cierre Fin de Mes (EOM)**: Truncamientos artificiales a fin de mes **estrictamente excluidos** de las métricas de selección de estrategias.
- **Aislamiento OOS**: Partición de prueba reservada (`TEST`) sellada de forma hermética (No TEST selection).
