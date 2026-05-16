# Checklist de Activación: LIVE_SANDBOX_100USD

Para proceder a la creación y activación del ejecutor real de 100 USD, todos los puntos deben estar en estado **VERDE**.

## A. Pre-requisitos Institucionales
- [ ] Veredicto del Gate: **`DEMO_TP_GATE_PASS`** obtenido.
- [ ] Informe de auditoría del trade demo revisado y firmado.
- [ ] Aceptación explícita del usuario del riesgo de 100 USD.

## B. Configuración Técnica
- [ ] Cuenta MT5 Real separada con balance de 100 USD (ni más, ni menos).
- [ ] Terminal MT5 Real configurado y logueado.
- [ ] Símbolo EURUSD seleccionado y visible.
- [ ] Magic Number para real definido (diferente al de demo).

## C. Seguridad y Riesgo
- [ ] Lote mínimo (0.01) verificado para EURUSD.
- [ ] Riesgo máximo en USD por trade calculado (< 0.25 USD).
- [ ] Kill Switch configurado para 10 USD de pérdida máxima.
- [ ] Filtro de noticias Fortress V3 actualizado y activo.

## D. Infraestructura
- [ ] Heartbeat configurado para la instancia real.
- [ ] Directorio de logs `results/live_sandbox/` creado.
- [ ] Telemetría de real aislada de la de demo/shadow.

---
**SI ALGUNA CASILLA ESTÁ VACÍA: PROHIBIDO ACTIVAR.**
