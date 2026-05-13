# AUDITORÍA PRE-EJECUCIÓN DE DATOS Y NOTICIAS (DATA/NEWS PRECHECK)

## 1. Disponibilidad y Cobertura
- **Dataset Canónico**: Archivo de eventos `news_eurusd_am_fortress_v3.csv` presente y accesible en la bóveda inmutable (`05_MARKET_DATA_VAULT/data/`).
- **Ventana Temporal de Cobertura**: Abarca de forma continua e ininterrumpida desde el `2020-01-01` hasta el `2026-04-30`, alineándose al milímetro con las particiones de la estrategia R1.

## 2. Continuidad y Zonas Horarias
- **Integridad de Series de Ticks**: Cero lagunas de datos o ausencias superiores a 5 días hábiles consecutivos en los parquets de la bóveda.
- **Alineación de Husos Horarios**: Conversiones estrictas de marcas de tiempo UTC hacia `America/New_York`, respetando las transiciones históricas automáticas de horario de verano (DST correcto verificado en la suite V7).

## 3. Barreras de Exclusión Institucional
- **Rollover y Spreads Extremos**: Ventana de liquidez reducida de `16:55` a `17:15` NY físicamente bloqueada para la toma de nuevas posiciones intradía.
- **Buffers de Noticias (Tier-1 Impact)**: Suspensión de la ejecución algorítmica en buffers pre y post evento configurados nativamente en el motor central.
- **Mecanismo Fail-Close**: Activado. Ante la indisponibilidad de lectura del CSV de noticias, el motor aborta la evaluación de la vela para evitar la toma ciega de riesgos.
