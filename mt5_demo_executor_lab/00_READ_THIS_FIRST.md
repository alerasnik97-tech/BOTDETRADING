# 00_READ_THIS_FIRST.md - MT5 Demo Executor Lab

## Propósito
Este laboratorio es un entorno de simulación técnica de alta fidelidad. Su objetivo es validar la **tubería técnica (piping)** entre el cerebro Python y el terminal MetaTrader 5 antes de cualquier consideración de trading real.

## Estado Institucional
- **Modo:** `DEMO_ONLY` / `SANDBOX`
- **Trading Real:** **ESTRICTAMENTE PROHIBIDO**.
- **Conectividad:** Python ↔ MT5 (Cuenta Demo).
- **Estrategia:** Candidato Shadow Seleccionado (SCBI_H1_M5_RECLAIM).

## Reglas de Seguridad
1. **Aislamiento Total:** Este código no debe importar ni modificar archivos del `research_lab` (producción).
2. **Hard Stop:** Si la cuenta detectada no es de tipo "demo" o "preliminary", el ejecutor debe abortar inmediatamente.
3. **Control de Riesgo:** Aunque sea demo, se aplican límites de 0.10% para forzar disciplina técnica.
4. **Kill Switch:** Se activa por 3 SL seguidos o 5% de DD acumulado en la cuenta demo.

## Componentes
- `mt5_demo_executor.py`: Orquestador del loop principal (Scan -> Decide -> Route).
- `mt5_data_bridge.py`: Conversión de ticks/rates de MT5 a objetos de datos compatibles con el lab.
- `mt5_order_router.py`: Gestión de órdenes, SL, TP y registro de fills.
- `mt5_timeout_manager.py`: Monitoreo activo de posiciones para cierre a las 4 horas.
- `mt5_news_guard.py`: Filtro de noticias basado en el dataset institucional.

---
**Este entorno NO es el bot de producción. Es un banco de pruebas para la ejecución automática.**
