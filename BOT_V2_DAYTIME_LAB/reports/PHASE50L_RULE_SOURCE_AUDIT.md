# Auditoría de Fuentes de Reglas (Phase 50L)

## 1. Resumen de la Discrepancia
Se ha detectado una "Alucinación de Reglas" en la implementación de la Fase 50K. Los parámetros fueron interpretados de forma incorrecta respecto a la autoridad oficial del proyecto.

| Parámetro | Interpretación Phase 50K (Errónea) | Interpretación Autoridad (Phase 25/27) |
| :--- | :--- | :--- |
| **BF (Body Filter)** | Break-even Factor (Trigger de BE al 70% del TP) | Body Filter (Filtro de entrada: cuerpo/rango >= 0.7) |
| **BE (Break-Even)** | BE Protegido (Mueve Stop a +0.4R) | BE Trigger (Trigger a 0.4R, mueve Stop a Entrada 0.0R) |
| **TP (Take Profit)** | 1.4R | 1.4R |

## 2. Evidencia en el Código de Autoridad

### [Phase 27 (Validación Histórica)](file:///C:/Users/alera/Desktop/Bot/BOT%20DE%20TRADING%20ultimo/BOT_V2_DAYTIME_LAB/src/phase27_full_historical_validation.py)
```python
# Línea 21: Configuración oficial
CONFIG = {"tp_r":1.4, "be_r":0.4, ..., "body_filter_pct":0.70}

# Línea 138: Lógica de BE (Trigger a be_r, Stop a entrada)
if row.high_bid >= active['entry_price'] + active['risk'] * config['be_r']:
    active['sl'] = active['entry_price'] # 0.0R
    active['be_triggered'] = True

# Línea 159: Lógica de BF (Filtro de entrada)
if body_pct > 0:
    body = abs(row.close_bid - row.open_bid)
    wick = row.high_bid - row.low_bid
    if wick > 0 and body/wick < body_pct: continue # Descarta el trade
```

### [Manipulante Config (Authority)](file:///C:/Users/alera/Desktop/Bot/BOT%20DE%20TRADING%20ultimo/MANIPULANTE/01_ESTRATEGIA_AUTORIDAD/manipulante_config.json)
```json
"be_r": 0.4,
"body_filter": 0.7,
```

## 3. Consecuencias Técnicas
1. **Trigger Tardío**: Phase 50K activaba el BE a **0.98R** (70% de 1.4) en lugar de **0.4R**.
2. **Stop Agresivo**: Phase 50K intentaba proteger **0.4R** de ganancia, lo que en un activo de alta volatilidad (ticks) provoca que muchos trades que hubieran llegado al TP sean cerrados prematuramente por ruido en el nivel 0.4R.
3. **Filtro Inexistente**: Phase 50K no aplicaba el filtro de cuerpo de vela en la entrada, permitiendo trades "débiles" que la autoridad hubiera ignorado.

## 4. Conclusión Preliminar
El resultado de Phase 50K (**PF 0.04**) no es representativo de MANIPULANTE. Es un resultado de una estrategia modificada ("Shadow Variant") con reglas de gestión de riesgo no validadas.
