# MANIFIESTO DE PRE-COMPROMISO METODOLÓGICO DEL CONFIRMATION GAUNTLET

## 1. Filosofía de la Prueba (Cero Ilusión)
El objetivo primordial de esta fase no consiste en desplegar barridos de fuerza bruta para aislar combinaciones espurias altamente rentables sobre datos históricos (*overfitting*), sino someter la vecindad de la configuración ganadora de la expansión (`cfg_r1_expansion_opt1`) a un estrés metódico para certificar si el *edge* reside de forma estable y robusta en todo el subespacio paramétrico.

## 2. Reglas Estrictas de Selección y Acreditación
- **Dimensionamiento Acotado**: El gauntlet explora de forma controlada una retícula de **324 permutaciones** en torno al ensamble dominante de V41.
- **Inviolabilidad de la Partición de Prueba**: El segmento de datos fuera de muestra (**TEST: 2025-01 a 2026-04**) permanece rigurosamente cegado y vedado para labores de selección, ajuste o filtrado.
- **Jerarquía de Criterios sobre TRAIN/VAL**:
  1. **Supervivencia de Costos**: $PF_{val\_net\_0.2} \ge 1.20$ neto de slippage físico y comisiones FTMO.
  2. **Consistencia Transaccional**: Volumen muestral de alta representatividad ($N_{val} \ge 50$).
  3. **Inmunidad a Quiebras**: Estado de cuenta FTMO intacto.
  4. **Estabilidad de Racha**: Drawdowns controlados que no pongan en riesgo el capital de trabajo.
- **Single-Run Definitivo**: El candidato o ensamble representativo que prevalezca tras los filtros rigurosos en validación será ejecutado **una única y última vez** sobre la muestra ciega `TEST`. Se prohíbe re-calibrar parámetros tras observar el resultado de dicha partición.
