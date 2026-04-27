# Reglas de Escalado - Freno a la Impulsividad

> [!CAUTION]
> **ESTADO: NOT_ACTIVE_UNTIL_MICRO_PILOT_ALLOWED**

El mayor riesgo en el trading real es el escalado prematuro basado en rachas de corto plazo. Este documento prohíbe el escalado automático.

## Prohibiciones de Escalado
- [ ] **NO existe la promoción automática:** Haber ganado 5 trades seguidos NO habilita a subir el riesgo.
- [ ] **NO al compounding:** Durante el micro piloto, las ganancias se mantienen en la cuenta pero NO se usan para aumentar el tamaño de lote.
- [ ] **NO al escalado emocional:** La sensación de "ya lo entiendo" o "el mercado está fácil" es un trigger para NO escalar.

## Condiciones para Solicitar Escalado (Post-Piloto)
Pasar de Micro Piloto a una fase superior requiere:
1. **N Real >= 30 trades** documentados y auditados.
2. **Slippage < 0.2 pips** promedio de desviación respecto a Shadow.
3. **Drawdown Máximo Real < 5.0%**.
4. **Nuevo Tribunal Institucional:** Se requiere un nuevo gate y una nueva revisión de todo el set de datos real.

## Regla de Protección
Cualquier intento de "acelerar" el proceso sin cumplir las métricas de evidencia resultará en la revocación inmediata del permiso de trading real y el retorno a Shadow por un periodo indefinido.

**Recuerda:** El objetivo es la robustez a largo plazo, no el pelotazo a corto plazo.
