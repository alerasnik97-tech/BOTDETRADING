
# PHASE 25: KILL SWITCH POLICY (PAPER DEMO)

## 1. BLOQUEOS AUTOMÁTICOS (FAIL-CLOSED)
- **News Fortress**: Si el estado es BLOCK o MISSING, no se opera.
- **Data Mask**: Si la calidad de data es insuficiente, no se opera.

## 2. DISPARADORES DE PAUSA TÉCNICA (MANUAL)
- **Error de Lógica**: Si se ejecuta un trade fuera de la ventana horaria (07:00-16:30 NY).
- **Error de Gestión**: Si aparece una operación sin SL o TP.
- **Falla Técnica**: 2 errores técnicos en una misma semana calendario.
- **Hash Mismatch**: Si la configuración es modificada sin auditoría.

## 3. DISPARADOR DE DRAWDOWN (RISK)
- **Pausa de Emergencia**: Si el Drawdown acumulado en demo alcanza **-6.0 R** (1.2x del máximo histórico auditado de -5.0R).

---
*Firma: PHASE25_RISK_AUTHORITY_SIG*
