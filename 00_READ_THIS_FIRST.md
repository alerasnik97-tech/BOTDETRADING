# 00 READ THIS FIRST - DOCUMENTO MAESTRO

Este es el documento más importante del proyecto. Cualquier IA o colaborador debe comenzar aquí para entender la jerarquía y el estado actual del bot.

## 1. Raíz Oficial y Única
La única raíz oficial del proyecto es:
`C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo`

**ESTÁ PROHIBIDO** usar carpetas externas como `Bot V1` o `Bot V2` o subcarpetas antiguas en la raíz. Toda la evidencia histórica y duplicada ha sido movida a `ARCHIVE_SUPERSEDED\`.

## 2. Jerarquía de Autoridad Estratégica

### 2.1 SCBI_M5_GLOBAL (Overnight)
- **Rol:** Estrategia principal y fuente de verdad.
- **Estado:** PROTEGIDA. No se modifica ni se adapta.
- **Uso:** Requiere infraestructura VPS para operar en la madrugada de Londres.

### 2.2 BOT_V2_DAYTIME_LAB (Diurno)
- **Rol:** Laboratorio de investigación diurna independiente.
- **Estado:** ACTIVO.
- **Advertencia:** No confundir resultados de laboratorio con autoridad de producción de SCBI.

## 3. Estado de Candidatos (Living Candidates)
Los únicos candidatos vivos para futura observación/forward en la familia diurna son:
- **Phase 7 Repaired:** Candidato balanceado (PF 1.50).
- **Phase 8 High Precision:** Candidato de alta calidad (PF 2.09).

## 4. Experimentos Rechazados
- **Phase 9 / 10 / 11:** RECHAZADOS. Han sido descartados por falta de robustez, destrucción de edge por frecuencia o falta de mejora sobre el baseline.

## 5. Reglas de Oro Operativas
1. **Nada está aprobado para real automático todavía.**
2. No usar reportes dentro de `ARCHIVE_SUPERSEDED` como autoridad.
3. Toda nueva investigación debe verificar:
   - BID/ASK/SPREAD real.
   - News Guard activo.
   - Protección contra Lookahead.
   - Costos y comisiones reales.
   - Robustez en el periodo 2023-2025.
4. El uso de Break-Even (BE) ha demostrado ser dañino en los modelos auditados.

## 6. Siguiente Paso Único
Establecer un **Forward Gate disciplinado** (Demo/Paper) para Phase 7 y Phase 8 antes de cualquier consideración de capital real.
