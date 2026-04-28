
# NEWS FORTRESS: POLÍTICA OPERATIVA INSTITUCIONAL

## 1. Filosofía de Seguridad
La prioridad absoluta de News Fortress es la **preservación del capital**. Preferimos perder 100 oportunidades de trading que sufrir una pérdida catastrófica por operar durante un evento de alta volatilidad sin protección.

## 2. Reglas de Bloqueo
El sistema bloqueará cualquier operación si:
- Existe una noticia de **Alto Impacto (High Impact)** en USD o EUR dentro de los ±60 minutos.
- Existe un evento **Ultra Crítico** (NFP, FOMC, CPI, Fed/ECB Rate) dentro de los ±120 minutos.
- Se detectan palabras clave de riesgo (Speech, Powell, Lagarde) incluso si el impacto marcado es menor.
- El calendario de noticias está desactualizado, vacío o no disponible (Fail-Closed).

## 3. Requerimientos de Ejecución
Ninguna orden puede ser enviada al mercado si:
- No tiene **Stop Loss (SL)** definido.
- No tiene **Take Profit (TP)** definido.
- El News Fortress Gate no responde explícitamente `ALLOW`.

## 4. Gestión de Fallos
Si el feed de noticias falla, el bot entra en modo **Cuarentena Total**. No se permite ninguna operación hasta que el administrador restaure el acceso a un calendario válido y certificado.
