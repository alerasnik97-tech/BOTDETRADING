
# PHASE 22 DAILY RUNBOOK (DEMO MODE)

## 1. PRE-SESSION (06:30 NY) - MANDATORIO
- **Config Hash Check**: Validar que la config no ha sido alterada.
- **Data Quality Gate**: Verificar que el feed de M3 no tiene gaps. Si la máscara da BLOCK, no se opera.
- **News Gate**: Verificar calendario. Si News Fortress no da ALLOW explícito, no se opera.
- **Connectivity**: Confirmar que el feed de Bid/Ask está activo.

## 2. SESSION (07:00 - 16:30 NY)
- **Signal Discovery**: Monitorear H1 Sweep + First M3 CHoCH.
- **Order Entry**: Ejecución manual o automática en cuenta DEMO únicamente.
- **Management**: Verificar activación automática de BE 0.5R. No intervenir manualmente.

## 3. POST-SESSION (17:00 NY)
- **Mandatory Close**: Cerrar cualquier trade abierto antes de las 20:00 NY.
- **Logging**: Registrar 0 violaciones de News Fortress y 0 de Data Mask.
- **Performance**: Actualizar diario de trades (Demo Ledger).
