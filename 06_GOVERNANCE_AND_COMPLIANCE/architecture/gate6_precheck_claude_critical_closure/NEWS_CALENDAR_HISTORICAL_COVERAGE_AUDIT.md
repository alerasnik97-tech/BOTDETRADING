# AUDITORÍA DE COBERTURA HISTÓRICA DEL CALENDARIO DE NOTICIAS

## 1. Integridad de Ingestión
El archivo canónico de la bóveda (`news_eurusd_am_fortress_v3.csv`) contiene eventos macroeconómicos de impacto alto para EUR y USD desde 2015 hasta 2026.
- Total de eventos cargados exitosamente por el parser: **1,106 eventos de alto impacto**.

## 2. Bloqueo Causal
El filtro de noticias `NewsCalendar` orquestó exitosamente la inserción de periodos de blindaje ortogonal (modo `post5` activo en la sonda) evadiendo la entrada a mercado en los primeros 5 minutos posteriores a cada evento crítico.
