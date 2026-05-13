# AUDITORÍA DE RESTRICCIONES E HIGIENE CAUSAL (FULL RUN CONSTRAINT AUDIT)

## 1. Verificación de Conformidad Institucional
Se certifica el cumplimiento absoluto y sagrado del 100% de las imposiciones del protocolo de ejecución durante el cómputo de los 76 meses:

- [x] **Activo Exclusivo**: `EURUSD` verificado en todas las tuplas transaccionales.
- [x] **Límite Intradía**: Operaciones de ingreso confinadas estrictamente al bloque `07:00-17:00` NY.
- [x] **Cuota Operativa**: Ningún exceso sobre `max_trades_per_day = 3` en el histórico transaccional.
- [x] **Veto de Rollover**: Cero ingresos de `16:55` a `17:15` NY (Bloqueo verificado).
- [x] **Filtros Macroeconómicos**: Buffers de exclusión temporal aplicados incondicionalmente en eventos Tier-1.
- [x] **Fail-Close**: Válvula de interrupción de lectura de noticias operando correctamente.
- [x] **Costos Redondos**: Deducción de comisiones FTMO implementada a nivel de lote.
- [x] **Deslizamiento Obligatorio**: Impacto contable de `0.2` pips de slippage en cada transacción de las curvas de capital netas.
- [x] **Causalidad de Precios**: Ejecuciones referenciadas al Ask para compras y al Bid para ventas.
- [x] **Causalidad de Barra ($T+1$)**: Señales gatilladas puramente en la apertura de la siguiente barra inmediata tras el evento de absorción.
- [x] **Inmunidad Contable de Truncamiento**: Incidencia nula ($0$) de cierres de simulación a fin de mes (`EOM`) en las métricas de selección.
- [x] **Integridad OOS**: Cero retroalimentación o filtrado condicional desde la muestra de prueba (`TEST`).
- [x] **Inmutabilidad de Fuentes**: Cero deriva en las firmas SHA256 del orquestador (`no runner drift`).
- [x] **Inmutabilidad de Motor**: Paridad institucional canónica retenida incondicionalmente (`no engine drift`).

## 2. Veredicto Final de Estado
**ESTADO DE AUDITORÍA: PASSED (Conformidad Absoluta).**
No existen desviaciones ni violaciones de restricciones que invaliden la serie de resultados.
