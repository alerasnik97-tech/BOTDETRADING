# REAL_READINESS_POLICY.md
## Política Institucional de Gates de Transición

**Estado:** V1.0 - Mandato Operativo
**Propósito:** Definir los criterios técnicos no negociables para la promoción de líneas de trading entre estados de madurez.

---

## 1. DEFINICIÓN DE ESTADOS

| Estado | Significado Institucional |
|--------|---------------------------|
| **RESEARCH_ONLY** | Fase de laboratorio. Sin ejecución forward. |
| **SHADOW_ONLY** | Ejecución forward en entorno aislado y ledger separado. |
| **REAL_ELIGIBLE** | Cumple todos los gates. Apto para consideración de piloto real controlado. |

---

## 2. BLOQUE A: ROBUSTEZ HISTÓRICA (GATES DUROS)

Para que una línea sea considerada para **SHADOW_ONLY**, debe superar estos umbrales en backtest (IS+OOS):

- **Sample Size Mínimo (N):** >= 1500 trades.
- **Profit Factor (PF) Mínimo:** >= 2.0 (Conservador).
- **Expectancy Mínima:** >= 0.30R por trade.
- **Max Drawdown Máximo:** <= 15.0R.
- **Year Positive Ratio:** 1.0 (100% de los años en la muestra deben ser positivos).
- **Estabilidad Multi-año:** Ningún año puede representar más del 30% del profit total (evita dependencia de outliers).
- **Consistencia OOS:** PF_OOS no debe degradarse más de un 20% respecto a PF_IS.

---

## 3. BLOQUE B: ROBUSTEZ OPERATIVA

- **Runner Reproducible:** El código de ejecución debe estar en `scratch/` o `scripts/` con control de versiones.
- **Namespace Aislado:** Obligatorio el uso de carpetas separadas para resultados (`results/shadow/`).
- **Integridad de Datos:** Cobertura de datos H1/M5/News validada al 100% para el período evaluado.
- **Telemetría:** Presencia de sidecar trace para observabilidad completa del lineage de cada trade.
- **Zero Errors:** Sin errores estructurales, excepciones no manejadas o cuellos de botella de I/O.

---

## 4. BLOQUE C: ROBUSTEZ DE RIESGO

- **Drawdown Contenido:** La curva de equidad no debe presentar "flat periods" superiores a 6 meses.
- **Sensibilidad a Noticias:** El News Fortress debe estar activo y auditado.
- **Regímenes Adversos:** Evaluación cualitativa de comportamiento en regímenes de alta volatilidad (e.g. 2020, 2022).
- **Concentración de Niveles:** Ningún tipo de nivel (Asia/London/PD) debe explicar más del 60% del edge total.
- **Timeout Dependency:** Menos del 40% de los trades deben cerrar por timeout (preferir TP/SL).

---

## 5. BLOQUE D: ROBUSTEZ FORWARD / SHADOW

- **Readiness for Shadow:** Definición clara de runner, ledger y telemetry sidecar.
- **Shadow Execution:** Para **REAL_ELIGIBLE**, se requiere un mínimo de N=20 trades en shadow line con drift < 10% vs baseline.
- **Consistencia de Comportamiento:** El comportamiento observado en forward/shadow debe ser consistente con la lógica IS/OOS (sin "sorpresas" de ejecución).

---

## 6. CRITERIOS DE CLASIFICACIÓN FINAL

### NOT_READY
- Falla cualquier gate del Bloque A o B.
- Bloqueo inmediato. Siguiente paso: Refinar hipótesis en Research.

### SHADOW_READY
- Pasa Bloques A, B y C.
- Autorizado para incubación en entorno aislado.
- NO es elegible para real.

### REAL_ELIGIBLE
- Pasa Bloques A, B, C y D (incluyendo evidencia de Shadow Execution).
- Permite iniciar el proceso de auditoría final para piloto real.

---

## 7. GOBERNANZA

- Este gate es **automático y duro**.
- No se permiten excepciones manuales basadas en "intuición".
- El veredicto del `evaluator.py` es la última palabra técnica.
