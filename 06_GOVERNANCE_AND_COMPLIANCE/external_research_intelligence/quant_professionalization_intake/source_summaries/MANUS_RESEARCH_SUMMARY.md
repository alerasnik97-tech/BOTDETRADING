# MANUS RESEARCH SUMMARY — V49.7

## Tesis Principal
La profesionalización requiere alineación con estándares institucionales (FIA) y cumplimiento estricto de las reglas de Prop Firms (FTMO), enfocándose en controles pre-trade y modularidad de componentes.

## Hallazgos por Categoría

### 1. Gestión de Riesgos (Estándares FIA)
- **Pre-Trade:** Implementar límites de tamaño de orden (fat-finger), tolerancia de precio y Kill Switches.
- **Post-Trade:** Conciliación diaria de posiciones para detectar discrepancias entre el bot y el broker.

### 2. Cumplimiento FTMO
- **Equity-based Risk:** El control de pérdida diaria (5%) debe basarse en el Equity flotante, no solo en el balance cerrado.
- **Midnight CET Reset:** La lógica de riesgo debe estar sincronizada con el horario del servidor FTMO (CET).

### 3. Infraestructura y Ejecución
- **Modularidad:** Separar Data Pipeline, Alpha Factory, Risk Engine y Execution Engine.
- **Conectividad:** Transicionar hacia FIX Protocol para mayor fiabilidad y menor latencia frente a las APIs de MT5.

### 4. Monitoreo
- Uso de Grafana/Prometheus para visualización de métricas en tiempo real.
- Alertas automáticas por canales externos (Telegram/Slack) ante fallos críticos.

## Recomendaciones de Implementación
- **Fase Inmediata:** Auditoría de cumplimiento de pérdida diaria basada en Equity.
- **Fase Media:** Desarrollo de un Risk Engine independiente (independiente de la estrategia).
- **Fase Avanzada:** Contenedorización de módulos y despliegue en nube con observabilidad industrial.
