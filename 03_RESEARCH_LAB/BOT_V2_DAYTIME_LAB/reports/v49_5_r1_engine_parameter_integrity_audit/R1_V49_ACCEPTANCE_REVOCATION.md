# RESOLUCIÓN DE REVOCACIÓN DE ACEPTACIÓN INSTITUCIONAL
**Estrategia:** R1 (Mean Reversion / NY Open Absorption)  
**Nivel de Autoridad:** JUNTA DIRECTIVA DE GOBIERNO Y COMPLIANCE  
**Estatus de Decisión:** VINCULANTE / INMEDIATO  

---

## 1. Revocación Contractual de Aceptación
Por medio de la presente resolución, se declara la **REVOCACIÓN TOTAL E IRREVOCABLE** del estatus de aceptación otorgado a las configuraciones finalistas de la estrategia R1 en el hito de entrega V49. Toda certificación previa de preparación o idoneidad queda sin efecto legal ni probatorio dentro del ecosistema del proyecto.

## 2. Inhabilitación de Fronteras Operativas
Queda expresamente prohibido y tipificado como falta grave de autoengaño cuantitativo la ejecución de cualquiera de las siguientes transiciones operativas:
- **NO SE AUTORIZA** la apertura, lectura o testeo sobre la partición fuera de muestra reservada **TEST (2025-2026)**.
- **NO SE AUTORIZA** el despliegue en entornos de simulación en vivo (Paper Trading).
- **NO SE AUTORIZA** la conexión con cuentas de demostración (Demo).
- **NO SE AUTORIZA** la postulación a evaluaciones de capital de terceros (Pruebas de Fondeo FTMO).
- **NO SE AUTORIZA** la asignación de capital institucional o ejecución en cuentas reales (Live/Real).

## 3. Fundamentación Exhaustiva de la Revocación
La anulación del pase de compuerta se sustenta en la concurrencia probada de los siguientes 5 vicios metodológicos y estructurales:
1. **Rowcount Mismatch:** Falsedad física en la declaración del tamaño muestral transaccional del Batch 3, encubriendo el escaneo de 4,895 operaciones bajo un reporte escalar truncado de 2,080.
2. **Selección In-Sample Contaminada por VAL:** Promoción de candidatos basándose exclusivamente en picos de rentabilidad durante la validación, ignorando pérdidas sistemáticas contundentes en la muestra de entrenamiento nativa ($PF_{train} < 1.0$).
3. **Duplicados Paramétricos (Redundancia):** Emisión de secuencias de operaciones idénticas a nivel de byte por parte de configuraciones con distintos modelos de entrada y stop loss, demostrando colisión de grilla o ignorancia de variables por el motor.
4. **Concentración Temporal de Rentabilidad:** Ausencia de robustez distribuida. El rendimiento neto depende de forma crítica de un único mes atípico, evidenciando fragilidad ante cambios de régimen.
5. **Estrés de Slippage Incompleto:** Carencia de auditorías de degradación bajo fricciones realistas de $0.3$ y $0.5$ pips aplicadas sobre el universo de finalistas depurados.
