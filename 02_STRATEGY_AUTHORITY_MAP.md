# STRATEGY AUTHORITY MAP

Este documento define la jerarquía y el estado de autoridad de todas las estrategias dentro del ecosistema del BOT.

## 1. Autoridad Máxima: SCBI_M5_GLOBAL (Overnight)
- **Estado:** PRODUCCIÓN / MADRUGADA.
- **Rango Operativo:** 00:00 - 05:00 UTC.
- **Benchmark:** Superior. No se modifica. No se adapta.
- **Infraestructura:** Requiere VPS o entorno 24/5. 
- **Nota:** Es la estrategia principal. No está reemplazada por Bot V2.

## 2. Candidatos Diurnos Certificados
- **Phase 8 High Precision**: PF 2.09. Estado: **VALIDADO**.
- **Phase 13 London Reclaim**: PF 1.62. Estado: **VALIDADO**.
- **Phase 17 Post-News**: PF 2.03. Estado: **VALIDADO**.
- **Phase 18 H1 Fractal Sweep**: PF 1.63. Estado: **VALIDADO_FOR_FORWARD**.

## 3. Candidato Diurno Balanceado: Phase 7 Repaired
- **Estado:** VALIDADO (PF 1.50).

## 4. Laboratorio Diurno: BOT_V2_DAYTIME_LAB
- **Estado:** LABORATORIO / INVESTIGACIÓN.

## 5. Experimentos Rechazados (Sin Autoridad Operativa)
- **Phase 9, 10, 11**: Rechazadas por baja robustez.
- **Phase 12**: **INVALIDADA** (Bug crítico detectado).

## 6. Validación de Infraestructura (Safety Gate)
- **Engine Safety Suite:** **APROBADO**.
- **Phase 18 Forensic Audit:** **PASSED** (No-lookahead certified).

## 7. Preparación para Despliegue (VPS Readiness)
- **Estado:** **VPS-READY**.
- **Gate de Control:** Forward Demo activo para fases 8, 13 y 18.

## 7. Regla de Oro
Ninguna estrategia fuera de esta lista tiene autoridad operativa. Los reportes obsoletos en `ARCHIVE_SUPERSEDED\` no deben ser usados para tomar decisiones.

## 8. Integridad de Directorios
- **Raíz Única:** `C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo`
- **Limpieza:** Todas las versiones anteriores han sido archivadas internamente.
