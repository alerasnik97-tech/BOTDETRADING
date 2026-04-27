# Parameter Tuning Roadmap: Wave 1 & 2 Optimization

Este documento detalla el plan estratégico de optimización para las estrategias que mostraron potencial inicial durante el estudio consolidado de ventanas. El objetivo es refinar los umbrales de entrada y salida para maximizar el Profit Factor OOS sin caer en el sobreajuste (overfitting).

## 1. Estrategia Prioritaria: zscore_mean_reversion_pm
Fue la única estrategia en pasar el protocolo de rechazo con un PF > 1.0.

### Hallazgos del Estudio
- **Ventana Óptima:** 16:30 NY.
- **Limitación Actual:** El edge es muy delgado (PF 1.01).
- **Hipótesis:** El umbral de Z-Score actual (2.0) captura demasiadas señales ruidosas.

### Plan de Tuning
| Parámetro | Rango Sugerido | Justificación | Prioridad |
|-----------|----------------|---------------|-----------|
| `zscore_threshold` | 2.2, 2.5, 2.8, 3.0 | Filtrar señales de menor probabilidad y buscar reversiones en extremos reales. | Crítica |
| `target_rr` | 1.0, 1.2, 1.5 | Al ser reversión, un RR menor puede aumentar el Win Rate y la estabilidad. | Alta |
| `atr_filter` (nuevo) | 0.5 - 2.0 | Evitar entradas en periodos de volatilidad muerta. | Media |

---

### 2. Candidatas a Rescate (Wave 2)
Estrategias que fallaron por poco o por errores de convención ya reparados.

#### triple_macd_filter
- **Falla original:** No evaluada por bug de parámetros.
- **Próximo Paso:** Optimizar los periodos de MACD (específicamente la relación entre el fast_hist y el main_hist).
- **Rango:** Probar combinaciones de señales rápidas (6/13/5) vs lentas (24/52/18).

#### turtle_soup_fade
- **Falla original:** PF OOS < 1.0.
- **Próximo Paso:** Incrementar el `lookback` Donchian de 20 a 55 bars para capturar rupturas de niveles más significativos en HTF.

---

### 3. Matriz de Volatilidad (Filtro Global)
Se propone introducir un "Regime Filter" en los próximos backtests para todas las estrategias:
- **Parámetro:** `shock_candle_atr_max`.
- **Rango:** 1.5 a 3.0.
- **Objetivo:** Determinar si filtrar velas excesivamente grandes (noticias) mejora el PF en la sesión PM.

## Resumen de Prioridad de Optimización
1. **Z-Score Mean Reversion:** Optimización agresiva de umbrales.
2. **ICT FVG (Incubada):** Definición de primer Grid de búsqueda.
3. **Triple MACD:** Auditoría de señales de momentum.

> [!TIP]
> Para la sesión PM, la **paciencia** es el edge. Generalmente, aumentar los umbrales de entrada (ser más exigente) mejora los resultados a costa de reducir el número de operaciones.
