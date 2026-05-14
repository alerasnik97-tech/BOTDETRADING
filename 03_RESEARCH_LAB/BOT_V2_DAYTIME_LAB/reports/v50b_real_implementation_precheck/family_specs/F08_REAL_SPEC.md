# F08 REAL SPEC ?" Session Overlap Trend

**Hiptesis**: El solapamiento entre Londres y Nueva York (08:00 ?" 11:00 NY) es el periodo de mayor liquidez y volumen, ideal para capturar la tendencia dominante del da.

## Condicin de Seİal
- **Trigger**: Cruce de EMAs (ej: 9 y 21) en timeframe de 5m durante la ventana de solapamiento.
- **Filtro**: El precio debe estar por encima/debajo del VWAP diario.

## Parǭmetros Micro
- **Stop**: Swing High/Low reciente.
- **Target**: 2.0 R.
- **Max Trades/Day**: 2.

## Diferencia vs R1
- R1 no utilizaba el VWAP ni el solapamiento de sesiones como ancla de liquidez.
