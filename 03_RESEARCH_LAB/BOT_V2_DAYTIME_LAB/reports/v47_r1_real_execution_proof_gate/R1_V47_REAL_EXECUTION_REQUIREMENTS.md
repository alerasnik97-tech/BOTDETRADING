# REQUISITOS DE EJECUCIÓN REAL — R1 V47

## 1. Infraestructura de Prueba
- **Motor**: UnifiedV7Engine (src/v7_engine/engine.py).
- **Datos**: Tick data real (05_MARKET_DATA_VAULT/BOT_MARKET_DATA/tick/EURUSD/monthly).
- **Entorno**: venv_v37 con pandas, numpy, pyarrow.

## 2. Parámetros del Micro-Run
- **Instrumento**: EURUSD.
- **Período**: 2025-01-01 a 2025-01-31 (Muestra real TEST).
- **Configuración**: `cfg_r1_expansion_opt1` (Parámetros derivados de V42).
- **Slippage**: 0.2 pips netos.

## 3. Criterios de Éxito
- Generación de trades con timestamps de microsegundos.
- Precios con precisión de tick real.
- PnL calculado por el `CostModel` institucional.
- Engine Verify OK antes y después de la corrida.
