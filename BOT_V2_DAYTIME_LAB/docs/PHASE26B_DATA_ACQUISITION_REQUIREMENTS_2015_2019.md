# PHASE 26-B: DATA ACQUISITION REQUIREMENTS 2015-2019

## REQUERIMIENTOS DE DATA
- Símbolo: EURUSD.
- Período: 2015-01-01 a 2019-12-31.
- Fuente ideal: Dukascopy (o equivalente).
- Timeframe base: M1 BID/ASK real o Tick BID/ASK real.
- PROHIBIDO: M5, H1, MID-only, synthetic ticks, interpolación, forward-fill, derivar M3 de M5.

## CAMPOS ESPERADOS
- Timestamp
- Bid Open/High/Low/Close o Tick Bid
- Ask Open/High/Low/Close o Tick Ask
- Volume (opcional)

## PROTECCIÓN DE ENTORNO
La data en crudo NO debe incluirse dentro de los archivos ZIP canónicos debido a su peso. Debe generarse hashes SHA256 para probar su integridad. La Phase 25 permanece aislada.
