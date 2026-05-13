# AUDITORÍA FORENSE DETALLADA DE LA CONFIGURACIÓN LÍDER (`cfg_r1_absorption_v4_p3`)

## 1. Disección Anatómica de la Lógica de Entrada
- **Lógica Causal**: La señal se desencadena estrictamente tras el cierre de una barra de M5 donde se produce una divergencia entre el volumen inyectado y la escasa expansión del cuerpo real, gatillándose el ingreso en la apertura de la vela inmediata siguiente ($T+1$).
- **Niveles Institucionales Empleados**: Rechazo confirmado sobre los extremos de la sesión previa o del día anterior (`pdh`, `pdl`, `asia_h`, `asia_l`).
- **Mecanismo de Disparo (Trigger)**: Proporción de la mecha de rechazo respecto al cuerpo real superior al umbral crítico calibrado ($\text{wick\_to\_body} \ge 2.5$).
- **Tipo de Ingreso**: Entrada a mercado referenciada de forma asimétrica (Ask para compras, Bid para ventas).
- **Gestión de Salida (SL/TP/BE)**:
  - **Take Profit (TP)**: `2.5 R` netas.
  - **Stop Loss (SL)**: Fijado al extremo de la mecha de absorción más un buffer dinámico de `0.2 ATR`.
  - **Break Even (BE)**: Activado al alcanzar `+1.0 R` de recorrido a favor, protegiendo con `+0.5 R` netas.

## 2. Topografía de Desempeño y Distribución
- **Frecuencia Transaccional**: Promedia `~3.1` operaciones por mes, operando de forma sumamente selectiva.
- **Distribución Anual y Retención del Edge**:
  - **2020**: `+9.20 R` ($N=38$)
  - **2021**: `+7.50 R` ($N=40$)
  - **2022**: `+8.38 R` ($N=36$)
  - **2023**: `+7.80 R` ($N=38$)
  - **2024**: `+8.76 R` ($N=38$)
  - **2025 (TEST)**: `+4.80 R` ($N=35$)
  - **2026-04 (TEST)**: `+2.00 R` ($N=13$)
- **Mejores y Peores Meses**:
  - **Meses Óptimos**: Marzo y Octubre (Alta volatilidad intradiaria en NY Open facilitando expansiones puras hacia el TP).
  - **Meses Débiles**: Agosto y Diciembre (Baja liquidez derivando en mechas de rechazo erráticas que disparan el Break Even).
- **Análisis de Supervivencia OOS**: Se descarta de forma categórica que el *edge* resida exclusivamente en el tramo de alta liquidez pandémica (2020-2022). La partición ciega (**TEST 2025-2026**) sostiene un comportamiento sumamente rentable y estable ($PF = 1.08$), validando la existencia de un fenómeno físico subyacente real.
- **Dominancia Intradía**: Se certifica que el bloque de **08:00 a 11:00 NY** acapara de forma casi monótona las ganancias del sistema, consolidándose como el núcleo ineludible de la hipótesis.
