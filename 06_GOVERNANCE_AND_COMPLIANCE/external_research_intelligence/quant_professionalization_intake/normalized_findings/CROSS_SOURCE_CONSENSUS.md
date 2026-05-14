# CROSS-SOURCE CONSENSUS — V49.7

## Temas con Consenso Total (3+ fuentes sugeridas/identificadas)

### 1. Independencia del Módulo de Riesgo
- **Consenso:** El riesgo no debe ser parte de la "estrategia", sino un componente guardián independiente (FIA Standards).
- **Acción:** Crear un `RiskManager` que valide cada señal antes de enviarla al `Executor`.

### 2. Realismo de la Simulación
- **Consenso:** El backtest actual es demasiado optimista al ignorar slippage y latencia real.
- **Acción:** Integrar modelos de slippage basados en volatilidad y spreads variables de tick data.

### 3. Reproducibilidad Técnica
- **Consenso:** El acoplamiento con Windows y rutas locales frena la escalabilidad.
- **Acción:** Transicionar a un entorno basado en contenedores (Docker) y rutas relativas.

## Contradicciones Identificadas
- **FIX vs MT5:** Manus sugiere migrar a FIX Protocol para profesionalismo, mientras que la estructura actual está fuertemente ligada a MT5 (MQL5/Python API). La contradicción se resuelve manteniendo MT5 para la fase retail/prop-firm inicial y evaluando FIX para la fase de gestión de capital propio/institucional.

## Recomendaciones Prematuras (No hacer ahora)
- **Kubernetes:** Demasiado complejo para la escala actual del proyecto.
- **Microservicios masivos:** El overhead de red podría ser contraproducente antes de tener una cartera de 5+ estrategias activas.

## Recomendaciones Críticas (Hacer ASAP)
- **Equity-based Daily Loss Control:** Fundamental para no ser expulsado de FTMO por picos de volatilidad intradía.
- **Look-ahead Bias Harness:** Un error aquí invalida todo el research previo.
