# Checklist de Activación del Micro Piloto

Antes de ejecutar el primer trade real, se deben marcar todas las casillas:

## 1. Prerrequisitos de Infraestructura
- [ ] Runner productivo verificado (idempotencia y conectividad).
- [ ] Shadow Autopilot OK en las últimas 5 sesiones.
- [ ] Dataset de cobertura H1/M5/News al día.

## 2. Prerrequisitos de Gobernanza
- [ ] Veredicto del Micro Pilot Gate: `MICRO_PILOT_ALLOWED`.
- [ ] Checklist de riesgo ultra-conservador revisada.
- [ ] Kill Switch memorizado y accesible.

## 3. Revisión Diaria (Protocolo de Pilotaje)
- [ ] ¿Hay noticias de alto impacto en los próximos 30 min?
- [ ] ¿El Shadow Autopilot del día anterior dio señal coherente?
- [ ] ¿El saldo de la cuenta está dentro de los límites del piloto?

## 4. Procedimiento de Pausa
- [ ] En caso de duda técnica: Detener proceso y volver a `SHADOW_ONLY`.
- [ ] En caso de breach de riesgo: Activar Kill Switch y documentar.
