# PAPER TRADING ENTRY CRITERIA

Para que una estrategia sea candidata a Paper Trading / Incubación, debe cumplir estrictamente con los siguientes criterios técnicos y estadísticos:

## Criterios Operativos
- **Símbolo**: EURUSD únicamente.
- **Horario NY**: Máximo 07:00 - 17:00 NY.
- **Frecuencia**: Máximo 3 trades por día.
- **Instrumentos**: Solo activos permitidos en la configuración de la fase.

## Criterios de Performance (Netos)
Las métricas deben ser calculadas incluyendo comisión FTMO y slippage conservador (0.2 pips mínimo).

- **Profit Factor Validation (PF_val_net)**: >= 1.15.
- **Profit Factor Test (PF_test_net)**: >= 1.00.
- **Expectancy**: Neta positiva (en R o pips).
- **Drawdown**: Dentro de los límites de riesgo FTMO (Max Daily < 5%, Max Total < 10%).

## Criterios de Robustez
- **News Fail-Close**: Módulo activo y verificado.
- **Rollover**: Controlado, sin operaciones abiertas en cierre de sesión si aplica.
- **Trade Count**: Muestra estadística suficiente (mínimo 100 trades en backtest histórico).
- **Profit Concentration**: Sin dependencia extrema de < 5% de los trades.
- **No TEST Selection**: Los parámetros no fueron optimizados sobre el set de Test.

## Documentación Requerida
- Reporte final de Research aprobado.
- Archivo de parámetros final (frozen).
- Manifest de auditoría de datos.
- **Aprobación Explícita del Usuario**.

> [!IMPORTANT]
> Un backtest positivo NO habilita fondeo ni trading real. La incubación es un paso obligatorio e independiente.
