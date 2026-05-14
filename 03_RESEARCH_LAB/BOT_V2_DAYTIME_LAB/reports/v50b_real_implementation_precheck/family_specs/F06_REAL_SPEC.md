# F06 REAL SPEC ?" Volatility Regime Breakout

**Hiptesis**: Expansiones de volatilidad tras periodos de contraccin extrema (regimenes de baja volatilidad) generan movimientos tendenciales de alta probabilidad.

## Condicin de Seİal
- **Trigger**: Cierre de barra por fuera de Bandas de Bollinger (20, 2) o Keltner Channels cuando el ATR estǭ en el percentil inferior (25%) de los ltimos 100 periodos.
- **Ventana**: 08:00 ?" 11:00 NY.

## Parǭmetros Micro
- **Stop**: ATR * 1.5.
- **Target**: 2.5 R.
- **Max Trades/Day**: 1.

## Diferencia vs R1
- R1 no discriminaba por regmen de volatilidad previa.
