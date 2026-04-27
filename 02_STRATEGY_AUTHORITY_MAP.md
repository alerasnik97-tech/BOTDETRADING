# STRATEGY AUTHORITY MAP

Este documento define la jerarquía y el estado de autoridad de todas las estrategias dentro del ecosistema del BOT.

## 1. Autoridad Máxima: SCBI_M5_GLOBAL (Overnight)
- **Estado:** PRODUCCIÓN / MADRUGADA.
- **Rango Operativo:** 00:00 - 05:00 UTC.
- **Benchmark:** Superior. No se modifica. No se adapta.
- **Infraestructura:** Requiere VPS o entorno 24/5. 
- **Nota:** Es la estrategia principal. No está reemplazada por Bot V2.

## 2. Candidato Diurno Alta Precisión: Phase8 High Precision
- **Estado:** VALIDADO (Baja Frecuencia).
- **Profit Factor:** 2.09 aprox.
- **Ubicación:** `STRATEGIES\PHASE8_HIGH_PRECISION\`
- **Uso:** Ideal para cuentas que priorizan la calidad extrema sobre la cantidad de trades. Candidato para forward/demo.

## 3. Candidato Diurno Balanceado: Phase7 Repaired
- **Estado:** VALIDADO (Media Frecuencia).
- **Profit Factor:** 1.50 aprox.
- **Ubicación:** `STRATEGIES\PHASE7_REPAIRED_BALANCED\`
- **Uso:** Candidato principal para forward testing balanceado. Mayor frecuencia que Phase 8.

## 4. Laboratorio Diurno: BOT_V2_DAYTIME_LAB
- **Estado:** LABORATORIO / INVESTIGACIÓN.
- **Propósito:** Desarrollo de estrategias diurnas independientes.
- **Advertencia:** Separado totalmente de SCBI. No confundir resultados de laboratorio con autoridad de producción.

## 5. Experimentos Rechazados (Sin Autoridad Operativa)
Las siguientes fases han sido descartadas tras pruebas exhaustivas:
- **Phase 9 (Frequency Expansion):** Rechazada. Subir frecuencia destruyó el edge al relajar filtros.
- **Phase 10 (High Frequency Discovery):** Rechazada. No se encontró edge con 15–20 trades/mes y PF > 1.50.
- **Phase 11 (New Entries & Mgmt):** Rechazada. Nuevos métodos de gestión y BE no superaron a Phase 7/8.
- **Phase 12 (Benchmark Surpass):** **INVALIDADA**. Reportó PF 11.71 falso por bug de target invertido y omisión de spread.

## 6. Validación de Infraestructura (Safety Gate)
- **Engine Safety Suite:** **APROBADO**. El motor de simulación ha superado 16 tests de integridad (Bid/Ask, Math, Lookahead).
- **Autoridad Técnica:** No se aceptan métricas de laboratorio sin reporte `ENGINE_SAFETY_GATE_PASSED`.

## 7. Regla de Oro
Ninguna estrategia fuera de esta lista tiene autoridad operativa. Los reportes obsoletos en `ARCHIVE_SUPERSEDED\` no deben ser usados para tomar decisiones.

## 8. Integridad de Directorios
- **Raíz Única:** `C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo`
- **Limpieza:** Todas las versiones anteriores han sido archivadas internamente.
