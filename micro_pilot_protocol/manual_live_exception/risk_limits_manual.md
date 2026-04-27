# Límites de Riesgo Duros (Manual Micro-Pilot)

Este documento define las fronteras innegociables de capital para la fase de excepción manual. Cualquier violación de estos límites activa el **Kill Switch Manual**.

## 1. Límites Operativos
- **Riesgo por Trade:** 0.10% (mínimo) a 0.25% (máximo) del balance de la cuenta.
- **Máximo de Trades por Día:** 1 (Independientemente del resultado).
- **Máximo de Posiciones Simultáneas:** 1.
- **Riesgo Mínimo en Pips:** 2.0 pips (evitar micro-stops por ruido).

## 2. Límites de Pérdida (Drawdown)
- **Stop Diario:** 1.0% (Bloqueo inmediato si se alcanza por errores de ejecución o slippage).
- **Stop Semanal:** 2.5%.
- **Stop del Piloto:** 5.0% (Cierre definitivo de la fase de excepción).

## 3. Prohibiciones Estrictas
- **PROHIBIDO** escalar el tamaño de la posición tras una victoria (anti-compounding).
- **PROHIBIDO** subir el riesgo por "confianza" en un setup.
- **PROHIBIDO** cambiar parámetros (SL, TP, Buffers) en caliente durante un trade.
- **PROHIBIDO** operar cualquier otra estrategia o variante en la misma cuenta.
- **PROHIBIDO** el "revenge trading" o intentar recuperar pérdidas en el mismo día.

## 4. Gestión de Capital
- El balance base se fija al inicio de la semana.
- Las ganancias no se retiran hasta el fin del piloto.
- Las pérdidas se asumen como costo de validación.

---
> [!IMPORTANT]
> **La disciplina en el riesgo es el único objetivo de esta fase.** El profit es secundario.
