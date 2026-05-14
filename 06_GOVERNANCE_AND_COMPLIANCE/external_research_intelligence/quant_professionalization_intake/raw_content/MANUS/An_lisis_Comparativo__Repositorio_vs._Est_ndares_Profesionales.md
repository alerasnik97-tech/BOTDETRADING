# Análisis Comparativo: Repositorio vs. Estándares Profesionales

| Característica | Estado en el Repositorio | Estándar Profesional / Fondos Quant | ¿Qué Falta? |
| :--- | :--- | :--- | :--- |
| **Gestión de Riesgos** | Simulador de supervivencia FTMO, controles básicos de drawdown. | Pre-trade Risk Controls (Fat-finger, Price Tolerance, Kill Switches). | Implementar un motor de riesgo independiente que intercepte órdenes antes de MT5. |
| **Infraestructura de Datos** | Scripts para tick-by-tick y noticias. | Data Pipeline centralizado con limpieza, normalización y almacenamiento en DB (SQL/NoSQL). | Migrar de archivos CSV/JSON a una base de datos robusta (PostgreSQL/TimescaleDB). |
| **Ejecución** | MT5 Demo Executor Lab. | FIX Protocol, conectividad directa a exchanges, gestión de latencia. | Transicionar a FIX Protocol para reducir latencia y mejorar la fiabilidad en cuentas grandes. |
| **Backtesting** | Forensic Backtest, validación cruzada. | Event-driven Backtesting con simulación de deslizamiento (Slippage) y latencia. | Incorporar modelos de impacto de mercado y fricción de liquidez realistas. |
| **Arquitectura** | Modular pero con muchos scripts sueltos en `scripts/`. | Microservicios o arquitectura de componentes desacoplados (Docker/Kubernetes). | Refactorizar scripts sueltos en una librería core instalable y usar contenedores. |
| **Monitoreo** | Telemetría básica en `shadow_line_lab`. | Dashboards en tiempo real (Grafana/Prometheus) y alertas automáticas. | Implementar un sistema de monitoreo visual y alertas por Slack/Telegram para errores. |
| **CI/CD** | `bot_safety_ci.yml` básico. | Pipelines de CI/CD con tests de integración, stress tests y auditoría de código. | Aumentar la cobertura de tests unitarios y añadir tests de carga (Stress Testing). |
| **Cumplimiento (Compliance)** | Enfoque en reglas FTMO. | Auditoría completa, registros de auditoría inmutables, cumplimiento normativo. | Implementar un log de auditoría inmutable para cada decisión tomada por el algoritmo. |
