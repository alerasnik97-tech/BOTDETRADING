# V50A-QG FAMILY OBJECTIVE SPECS

## F01 ?" London Continuation
- **Hiptesis**: NY pullback es una trampa de liquidez; Londres retoma el control.
- **Ventana**: 08:30-11:00 NY.
- **Entrada**: Cruce de media rǭpida a favor de la tendencia de Londres tras pullback de 38.2%-61.8%.
- **Salida**: Fixed TP/SL o EOM (End of Month).
- **Invalidador**: Si Londres no tuvo tendencia clara (> 50 pips de rango) previo a NY.

## F06 ?" Volatility Regime Breakout
- **Hiptesis**: La volatilidad cicla entre contraccin y expansin.
- **Ventana**: 07:00-15:00 NY.
- **Entrada**: Ruptura de rango de 3 horas con ATR(14) < Percentil 25.
- **Salida**: Trailing Stop basado en ATR.
- **Invalidador**: Spread > 1.5 pips en el momento de la ruptura.

## F08 ?" Session Overlap Trend
- **Hiptesis**: El solapamiento genera el mayor volumen direccional.
- **Ventana**: 08:00-12:00 NY.
- **Entrada**: Ruptura de máximo/mnimo de la primera hora de NY a favor de Londres.
- **Salida**: Cierre de Londres (11:30-12:00 NY).
- **Invalidador**: Si hay noticias de alto impacto en los prximos 30 min.

## F12 ?" Macro Calendar Safe-Window
- **Hiptesis**: El mercado deriva tǸcnicamente sin ruido de noticias.
- **Ventana**: 09:00-15:00 NY (siempre que no haya noticias).
- **Entrada**: Patrn de barras de continuacin (Inside Bar) en M15.
- **Salida**: TP 2:1 o 16:00 NY.
- **Invalidador**: Cualquier noticia Medium/High en el calendario AM Fortress dentro de la ventana.
