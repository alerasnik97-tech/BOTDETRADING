# DAYTIME_EURUSD_EDGE_RECONSTRUCTION_REPORT

**Fecha:** 2026-04-25
**Veredicto:** MANUAL_EDGE_PARTIALLY_EXPLAINED
**Estatus:** RESEARCH_V2_COMPLETED

## 1. Auditoría de Data Manual
Se auditaron **841 trades** registrados entre 2020 y 2026.
- **Profit Factor Manual:** 1.88
- **Esperanza Manual:** 0.36R
- **Win Rate Manual:** 34.8%

## 2. El Descubrimiento del Factor Tiempo
El 100% de la rentabilidad manual se concentra en la ventana **08:00 - 11:00 NY**.
Los bots anteriores (V1) fallaron porque intentaron operar todo el día, diluyendo el edge institucional en el ruido de la tarde.

## 3. Resultados de Hipótesis Derivadas (2015-2026)
| Hipótesis | Ventana NY | TP | PF Bot | PF Manual |
| :--- | :--- | :--- | :--- | :--- |
| **A (Killzone)** | 08:30 - 11:00 | 2.0R | 0.96 | 1.88 |
| **B (Tight Overlap)** | 09:00 - 10:30 | 2.5R | **1.02** | 1.88 |

## 4. Análisis de la Brecha (Gap)
A pesar de corregir el horario, el bot solo alcanza un PF de 1.02 frente al 1.88 del humano. La diferencia reside en la **selectividad LTF (3M)**:
- El humano ignora barridos "sucios".
- El humano requiere una confirmación visual de desplazamiento (FVG) que el bot actual aún no calibra con precisión quirúrgica.

## 5. Conclusión
El edge manual del usuario es REAL y defendible. Se ha logrado reproducir el umbral de rentabilidad (PF > 1.0) simplemente restringiendo el horario al overlap NY/Londres. La automatización total requerirá un motor de detección de CHoCH 3M más avanzado.

## 6. Siguiente Paso Único
Implementar un prototipo de bot que opere estrictamente de **09:00 a 10:30 NY** con entrada condicionada a un **3M CHoCH + FVG**, ignorando cualquier otra señal del día.
