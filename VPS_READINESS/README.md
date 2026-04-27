# VPS READINESS PACKAGE

Este paquete contiene toda la infraestructura, documentación y scripts necesarios para desplegar el laboratorio de trading en una VPS (Virtual Private Server) Windows de forma segura.

## Propósito de la VPS
La VPS será utilizada inicialmente para:
- **Forward Testing / Demo:** Ejecución en tiempo real con cuentas de práctica.
- **Monitoreo:** Seguimiento constante del mercado EURUSD.
- **Estabilidad:** Garantizar uptime 24/5 para estrategias diurnas y futuras overnight.
- **Preflight:** Validación de entorno antes de cualquier escalada.

## Restricciones Críticas
- **PROHIBIDO TRADING REAL:** Esta configuración está bloqueada para `DEMO_ONLY`.
- **PROHIBIDA LA OPTIMIZACIÓN:** La VPS es para ejecución, no para investigación pesada.
- **PROHIBIDA LA MODIFICACIÓN DE REGLAS:** Las estrategias Phase 7 y Phase 8 están congeladas.

## Contenido del Paquete
- `VPS_SETUP_GUIDE.md`: Guía paso a paso de instalación.
- `scripts/`: Utilidades de validación y arranque seguro.
- `config_templates/`: Plantillas de configuración (sin secretos).
- `VPS_SECURITY_POLICY.md`: Normas de seguridad obligatorias.

---
**Veredicto Actual:** VPS_READY_FOR_DEMO_FORWARD_PREFLIGHT
