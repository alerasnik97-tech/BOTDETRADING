# V50B REAL QA ?" NEWS CALENDAR AUDIT

**Objetivo**: Evaluar la dependencia de F12 del calendario de noticias.

## Hallazgos
- El pre-check real us `DummyNews` (todas las noticias permitidas).
- F12 se basa en operar en ventanas "Safe macro", por lo que su validación actual es **débil**.
- No se han localizado archivos `.parquet` o `.json` de noticias en la ruta estándar del Vault (`05_MARKET_DATA_VAULT/BOT_MARKET_DATA/news`).

## Riesgos
- Sin noticias reales, el Performance Factor de F12 podría estar inflado o ser irreal al no filtrar periodos de alta volatilidad macro.

## Estado de F12
**F12_WITH_RESERVATIONS**

## Próximos Pasos
- Es mandatorio conectar el `NewsCalendar` real antes de declarar a F12 como "Listo para Fondeo".
- Si no hay noticias disponibles, F12 debe tratarse como una estrategia técnica pura sin filtro macro.

**Veredicto**: REAL_NEWS_NOT_WIRED_YET.
