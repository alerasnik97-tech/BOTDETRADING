# Investigación de Estándares de Trading Profesional y Reglas de Prop Firms

## 1. Controles de Riesgo Institucionales (Estándares FIA)
Los sistemas profesionales deben implementar controles pre-trade robustos:
- **Maximum Order Size (Fat-finger limits):** Evitar órdenes accidentalmente grandes.
- **Maximum Intraday Position:** Límite máximo de exposición neta por símbolo o total.
- **Price Tolerance:** Rechazar órdenes con precios demasiado alejados del mercado.
- **Cancel-On-Disconnect (COD):** Cancelación automática de órdenes si se pierde la conexión.
- **Kill Switches:** Capacidad de detener toda la actividad de trading instantáneamente.
- **Post-Trade Reconciliation:** Conciliación diaria de posiciones con el broker/exchange.

## 2. Reglas del Desafío FTMO (Prop Firm)
Para ser profesional en FTMO, el sistema debe cumplir estrictamente con:
- **Pérdida Diaria Máxima (5%):** Calculada sobre el capital (Equity), incluyendo pérdidas flotantes, comisiones y swaps. Se reinicia a medianoche CET.
- **Pérdida Máxima Total (10%):** El capital nunca puede caer por debajo del 90% del balance inicial.
- **Objetivo de Beneficio:** 10% en Fase 1 y 5% en Fase 2.
- **Días Mínimos de Trading:** 4 días por fase.
- **Apalancamiento Forex:** Hasta 1:100.
- **EAs/Bots:** Permitidos, pero con advertencias sobre el uso de EAs de terceros que puedan violar reglas de asignación de capital (copia de estrategias).

## 3. Arquitectura de Sistemas Quant Profesionales
Un sistema profesional debe ser modular:
- **Data Pipeline:** Gestión de datos históricos y en tiempo real (Tick-by-tick).
- **Alpha Factory:** Generación y backtesting de señales.
- **Portfolio Construction & Optimization:** Gestión de múltiples estrategias.
- **Risk Management Engine:** El núcleo que aplica los límites pre-trade.
- **Execution Engine:** Conectividad con brokers (MT5 API, FIX Protocol).
- **Monitoring & Alerting:** Dashboard de salud del sistema y alertas en tiempo real.
