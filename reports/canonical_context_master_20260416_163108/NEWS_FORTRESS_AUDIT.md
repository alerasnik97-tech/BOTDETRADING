# News Fortress Audit

## Classification

Veredicto global del motor de noticias:

`SAFE ONLY UNDER NARROW CONDITIONS`

## Why It Is Not Fully Reliable

- El dataset canonico general `data/news_eurusd_v2_utc_summary.json` sigue con `REJECTED_DISABLED`.
- Faltan familias criticas 08:30 en estado operacional usable, incluyendo NFP, unemployment y retail sales.
- Existe una debilidad explicitamente documentada para eventos EUR sensibles al gap de DST entre Europa y USA.
- El laboratorio tenia rutas fail-open: si la fuente quedaba deshabilitada, el research podia seguir con cero eventos y dar falsa sensacion de proteccion.

## Hardening Applied In This Cycle

- `validation.py` ahora falla cerrado si News Fortress esta habilitado pero la fuente no es operativa.
- `morning_challenge_runner.py` deja de marcar `news_filter_used=True` cuando la fuente no esta aprobada y pasa a exigir fuente operativa real.
- `light_runner.py` tambien pasa a fail-closed.
- `engine.py` ahora rechaza cualquier senal sin hard stop valido desde el origen.
- `engine.py` agrega trazabilidad real para exits `news_fortress_kill`.

## PM-Only Safe Scope

Se construyo un dataset derivado y estrecho:

- `data/news_eurusd_pm_research_safe.csv`
- Solo familias USD exactas y estables en NY: ISM 10:00, FOMC 14:00, FOMC press conference 14:30.
- Quedan excluidas todas las familias 08:30 y los eventos EUR con riesgo DST.

Para este scope puntual, el veredicto sube a:

`AUDITED AND SAFE FOR STRICT PM RESEARCH`

## 8:00 NY

Veredicto:

`NOT RELIABLE ENOUGH`

Motivos:

- La cobertura operacional AM sigue incompleta.
- La conclusion economica previa de "tecnicamente testeable pero esteril" hoy no alcanza como permiso para reabrir AM.
- Mientras no exista una fuente AM aprobada y fail-closed end-to-end, 8:00 NY no merece investigacion seria.
