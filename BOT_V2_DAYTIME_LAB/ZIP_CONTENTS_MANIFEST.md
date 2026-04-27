# ZIP_CONTENTS_MANIFEST - Bot V2

**Fecha de reconstrucción:** 2026-04-26
**Proyecto:** Bot V2 (Laboratorio de Research)
**Estado:** Reconstrucción Canónica — FASE 7 COMPLETADA

## Descripción
Este archivo detalla el contenido del paquete `000_PARA_CHATGPT.zip`, el cual representa el estado actual de la investigación en el laboratorio Bot V2, incluyendo el refinamiento estructural de la Fase 7.

## Compromisos de Integridad
- **Phase 7:** Se incluyen los resultados vigentes del refinamiento M3 First CHoCH (PF 1.64).
- **Integración VectorBT:** Si existe, se incluye como acelerador de research (no validador final).
- **No nuevas pruebas:** No se corrieron nuevas pruebas durante este empaquetado (solo empaquetado del estado actual).
- **No modificación:** No se modificaron métricas ni resultados previos.
- **Proyecto Principal:** No se tocó ni modificó el proyecto principal en `C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo`.
- **Canónico:** El único ZIP válido y actualizado para este laboratorio es `C:\Users\alera\Desktop\Bot\Bot V2\000_PARA_CHATGPT.zip`.

## Contenido Incluido
- `reports/`: Todos los informes MD y JSON (Fases 1 a 7).
- `outputs/`: Resultados de backtests, matrices de temporalidades, scoring y refinamientos de Fase 7.
- `src/`: Código fuente del motor de investigación (Phase6Engine) y orquestadores de refinamiento.
- `research_matrix/`: Hipótesis y matrices de decisión.
- `ZIP_CONTENTS_MANIFEST.md`: Este manifiesto actualizado.
- `VECTORBT_INTEGRATION_NOTES.md`: Notas sobre el uso de VectorBT como acelerador (si existe).

## Resultados Clave Incluidos
- **Phase 7 Master Candidate:** `strong_candidate_trades.csv` (PF 1.64).
- **Quality Matrix:** `quality_matrix_results.csv` (Fase 2).
- **Context Filters:** `exhaustion_refinement_results.csv` (Fase 7b).

## Exclusiones Técnicas
- Datasets pesados (.csv de precios, raw data).
- Cache y archivos temporales (__pycache__, .pyc).
- Git artifacts (.git).
- Entornos virtuales (.venv).
- ZIPs duplicados o antiguos.
