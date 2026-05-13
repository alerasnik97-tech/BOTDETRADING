# COMPARATIVA CONTRA BENCHMARKS DE LA EXPANSIÓN PARAMÉTRICA

## 1. Referencias Base y Aspiracionales
- **Manipulante Original (Manual)**: Desempeño aspiracional discrecional con Profit Factor bruto estimado de `~1.35 - 1.50`, normalizado sistemáticamente a `~1.10 - 1.20` tras imputar rigurosamente los costos físicos de comisiones y deslizamiento.
- **Piso Institucional**: Exigencia mínima de $PF_{val} \ge 1.15$ y $PF_{test} \ge 1.00$ para justificar viabilidad operativa.
- **Semilla Base de R1 (`cfg_r1_absorption_v4_p3`)**: $PF_{val\_net} = 1.18$, $PF_{test\_net} = 1.08$, Expectativa $= +0.18 R$ sobre $N = 238$ operaciones netas.

## 2. Matriz de Contraste del Ensamblado Óptimo en Expansión (`cfg_r1_expansion_opt1`)

| Métrica Crítica | Semilla Base R1 | Expansión Paramétrica V41 | Manipulante Original Normalizado |
| :--- | :--- | :--- | :--- |
| **Profit Factor Neto (VAL)** | 1.18 | **1.21** | ~1.10 - 1.20 |
| **Profit Factor Neto (TEST)** | 1.08 | **1.11** | N/A |
| **Expectativa Neta Global** | +0.18 R | **+0.21 R** | Estimada inferior en R puros |
| **Drawdown Máximo ($DD_r$)** | 3.40 R | **3.10 R** | Variable discrecional |
| **Frecuencia Operativa** | ~3.1 / mes | **~3.3 / mes** | ~10 - 15 / mes |
| **Ventana Operativa** | 07:00 - 17:00 NY | **08:00 - 11:00 NY** | Todo el día |

## 3. Certificaciones de Calificación
- **mejora vs base**: YES (Capitaliza un incremento de `+0.03` en el PF y reduce el Drawdown máximo al aislar el ruido vespertino).
- **supera piso institucional**: YES (Cumple de forma aplastante con las exigencias de rentabilidad).
- **se acerca al benchmark original**: YES (Alcanza e iguala el techo del desempeño teóricamente normalizado del modelo discrecional).
- **puede ser candidata de portafolio**: **YES** (La madurez y consistencia causal de la curva la acreditan como la primera piedra angular viable para la cartera de estrategias).
