# AUDITORÍA DE INTEGRIDAD DE PARÁMETROS DEL MOTOR (DIAGNÓSTICO V49.5)
**Estrategia:** R1 (Mean Reversion / NY Open Absorption)  
**Alcance Muestral:** EXCLUSIVAMENTE TRAIN/VAL (Sub-ventanas In-Sample: 2023-01 y 2024-06)  
**Veredicto de Integridad:** PARAMETER_NOT_HONORED_BLOCKER  

---

## 1. Diseño del Experimento de Aislamiento Paramétrico
Para comprobar de forma inequívoca si la capa de orquestación transaccional o la inyección de grillas (grid) padece de colisiones o ignora dimensiones clave, se diseñó un protocolo de estrés variacional de un solo factor (Ceteris Paribus). Se selecciona una configuración base representativa y se itera secuencialmente sobre sub-ventanas temporales conocidas por albergar alta densidad de señales de absorción institucional.

**Restricción de Cuarentena:** Se prohíbe de forma categórica y absoluta proyectar el escaneo hacia el intervalo 2025-2026 (TEST).

## 2. Matriz de Variación Diagnóstica (Ceteris Paribus)
Se aislaron tres vectores de control crítico, evaluando los deltas sobre los precios físicos de ejecución y las firmas de hash transaccional:

### A. Vector de Entrada (`entry_type`)
- **Variaciones:** `NEXT_OPEN` $\rightarrow$ `LIMIT_50_REJECTION` $\rightarrow$ `MIDPOINT_STOP`
- **Comportamiento Esperado:** Los precios de entrada (`entry_price`) y los instantes de activación (`entry_time`) deben diferir de forma determinista para reflejar los distintos modelos de microestructura intradiaria.
- **Hallazgo Físico:** **FALLIDO.** Múltiples permutaciones de `entry_type` arrojan bitácoras transaccionales idénticas a nivel de byte, conservando inalterados los hashes de conjunto de operaciones (`trade_set_hash`).

### B. Vector de Parada de Pérdidas (`sl_model`)
- **Variaciones:** `wick + 1.0 pip` $\rightarrow$ `wick + 1.5 pips` $\rightarrow$ `wick + 2.0 pips` $\rightarrow$ `microstructure + 1.5 pips`
- **Comportamiento Esperado:** Los precios de stop (`sl_price`) y las aserciones de riesgo asumido por transacción deben alterarse, modificando los retornos netos normalizados ($PnL_{net\_r}$).
- **Hallazgo Físico:** **FALLIDO.** Desensibilización extrema en subgrupos de la grilla. El runner orquesta salidas idénticas independientemente del modelo de mecha o microestructura invocado.

### C. Vector de Toma de Beneficios (`target`)
- **Variaciones:** `1.25R` $\rightarrow$ `1.5R` $\rightarrow$ `2.0R` $\rightarrow$ `2.5R`
- **Comportamiento Esperado:** Modificación estricta de las duraciones transaccionales y los precios de salida por límite (`exit_price`).
- **Hallazgo Físico:** Parcialmente honrado, pero supeditado a la colisión de los vectores de entrada y stop.

## 3. Emisión de Estatus Institucional
Se dictamina de forma indudable la existencia del estado de bloqueo:
$$\text{ESTADO} = \mathbf{PARAMETER\_NOT\_HONORED\_BLOCKER}$$

**Disposición de Higiene:** En estricto cumplimiento del mandato *NO TOCAR MOTOR*, este informe cumple un rol puramente diagnóstico. La arquitectura de `src/v7_engine/` y `src/v6_utils/` permanece intocada hasta que la junta de arquitectura autorice formalmente la apertura de un branch de remediación.
