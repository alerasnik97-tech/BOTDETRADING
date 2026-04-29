# MANIPULANTE — DAILY RUNBOOK

## Autoridad
**PHASE25** = TP 1.4R / BE 0.4R / BF 70% / EURUSD / M3 / H1 Fractal Sweep

## Ventana Operativa
- **Horario**: 07:00 – 16:30 NY
- **Max trades/day**: 1
- **Cierre obligatorio viernes**: 16:55 NY (REGLA GLOBAL — NO EXCEPCIONES)

## Pre-Mercado (06:30–07:00 NY)
1. Verificar News Fortress → debe ser ALLOW
2. Verificar Data Quality Mask → debe ser ALLOW
3. Si es viernes: confirmar que hay tiempo suficiente para cerrar antes de 16:55 NY
4. Verificar spread y condiciones de mercado
5. NO operar si hay duda

## Durante Sesión (07:00–16:30 NY)
1. Esperar H1 Fractal Sweep
2. Esperar First M3 CHOCH con body filter ≥ 70%
3. Ejecutar entrada con SL en sweep extreme + 0.5 pips buffer
4. TP = 1.4R
5. BE trigger = 0.4R
6. Registrar en ledger

## REGLA GLOBAL DE CIERRE VIERNES
**VIERNES 16:55 NY: CIERRE OBLIGATORIO**

- Cerrar CUALQUIER operación abierta antes de 16:55 NY
- No mantener posiciones durante el fin de semana
- No tomar señal si existe riesgo de no poder cerrar antes del cierre semanal
- No hay override manual
- Aplica a FTMO, FundedNext, paper/demo, free trial, y CUALQUIER prop firm
- Si hay duda → NO OPERAR

## Post-Sesión
1. Registrar resultado en dual ledger
2. Verificar compliance de todas las reglas
3. Si es viernes: confirmar 0 posiciones abiertas

## Kill Switch
- News Fortress no ALLOW → NO TRADE
- Data Quality Mask no ALLOW → NO TRADE
- Viernes sin margen para cerrar → NO TRADE
- Drawdown diario ≥ -2R → PAUSE
- Cualquier desviación manual → PAUSE
