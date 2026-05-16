# PROP FIRM COMPLIANCE SUMMARY

Este documento resume la compatibilidad de MANIPULANTE con las reglas comunes de las Prop Firms y cómo aseguramos el cumplimiento (Compliance) estricto de las mismas.

## Reglas de Cierre de Fin de Semana (Weekend Holding)
- **Riesgo Prop Firm**: Mantener posiciones el fin de semana viola las condiciones de la mayoría de evaluaciones de bajo costo y cuentas fondeadas (ej. FundedNext).
- **Compliance MANIPULANTE**: Solucionado mediante **GLOBAL_WEEKEND_HARD_CLOSE**. Los viernes a las 16:55 NY toda posición se liquida manualmente. 

## Reglas de Límite de Pérdida Diaria (Daily Loss Limit)
- **Riesgo Prop Firm**: Perder más de un 4% - 5% en un solo día causa la pérdida de la cuenta.
- **Compliance MANIPULANTE**:
  - Max Trades/Day = 1.
  - Riesgo recomendado = 0.50% por operación.
  - Es matemáticamente imposible perder más del 0.50% en un día de trading normal (incluso sumando un posible slippage moderado, nunca llegará al límite de la firma).

## Reglas de Noticias (News Trading)
- **Riesgo Prop Firm**: Algunas firmas restringen operar N minutos antes/después de noticias Red Folder.
- **Compliance MANIPULANTE**: Uso obligatorio de `News Fortress` en modo **Fail-Closed**. Si existe riesgo o cruce con eventos macro, no se permite abrir el trade.

## Consistencia (Consistency Rules)
- **Riesgo Prop Firm**: Requisito de que ningún trade o día represente más del X% de las ganancias totales.
- **Compliance MANIPULANTE**: Operativa de riesgo fijo (0.50%) y TP fijo (1.4R). Los retornos son totalmente lineales y consistentes.
