# V50A REJECTED FAMILIES

Familias descartadas en este ciclo y motivos de rechazo:

- **F04 (VWAP Mean Reversion)**: **Riesgo de Overfit Extremo**. Requiere demasiada optimizacin de parǭmetros de volumen y es altamente sensible a la calidad de los ticks agregados.
- **F09 (Microstructure Imbalance)**: **Costo Computacional e Incertidumbre**. Requiere anǭlisis de nivel de tick puro y es extremadamente sensible al slippage real del broker, difcil de modelar en backtest.
- **F10 (Failed Breakout Reversal)**: **Subjetividad y Correlacin**. Riesgo de parecerse demasiado a las lGicas de R1 (reversin) que ya demostraron ser dǸbiles.
- **F05 (Asian False Breakout)**: **Riesgo de Correlacin con Manipulante**. Aunque es vǭlida, muchas lGicas de limpieza de liquidez asiǭtica ya estǭn cubiertas por Manipulante. Buscamos diversificacin conceptual.
