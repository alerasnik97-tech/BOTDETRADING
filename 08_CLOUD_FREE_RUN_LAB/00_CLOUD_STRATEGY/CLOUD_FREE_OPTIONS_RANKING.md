# CLOUD_FREE_OPTIONS_RANKING

1. **Oracle Cloud Always Free**
   - **Para qué sirve**: VPS real (Ubuntu/Oracle Linux), ejecución persistente 24/7.
   - **Para qué NO sirve**: Cargas de CPU intensivas (ARM Ampere es bueno pero limitado), minado.
   - **Límite esperado**: 1-4 OCPUs, 6-24GB RAM (según región).
   - **Riesgo**: Suspensión si no hay actividad, dificultad para conseguir stock.
   - **Dificultad**: Media-Alta (configuración Linux/SSH).
   - **Overnight**: Sí.
   - **Tests**: Sí.
   - **Micro-probes**: Sí.
   - **Sweeps grandes**: No (CPU limitada).
   - **Recomendación**: Mejor opción para persistencia.

2. **Kaggle Notebooks**
   - **Para qué sirve**: Ejecución de scripts largos (hasta 12h-30h), GPU/TPU gratis.
   - **Para qué NO sirve**: Servidor persistente, entrada/salida interactiva de archivos compleja.
   - **Límite esperado**: 12h de sesión, 30h de GPU/semana.
   - **Riesgo**: Corte de sesión sin previo aviso.
   - **Dificultad**: Baja.
   - **Overnight**: Sí (si dura < 12h).
   - **Tests**: Sí.
   - **Micro-probes**: Sí.
   - **Sweeps grandes**: Sí (por bloques).
   - **Recomendación**: Ideal para sweeps de parámetros por su alta RAM.

3. **Google Colab Free**
   - **Para qué sirve**: Ejecución rápida, testing de lógica, análisis de datos.
   - **Para qué NO sirve**: Corridas nocturnas largas (desconexión por inactividad).
   - **Límite esperado**: Sesiones volátiles, RAM variable.
   - **Riesgo**: Desconexión frecuente.
   - **Dificultad**: Muy Baja.
   - **Overnight**: No.
   - **Tests**: Sí.
   - **Micro-probes**: No confiable.
   - **Sweeps grandes**: No.
   - **Recomendación**: Solo para prototipado rápido de lógica.

4. **GitHub Actions**
   - **Para qué sirve**: CI/CD, Pytest, validación de hashes, empaquetado.
   - **Para qué NO sirve**: Backtesting, procesamiento de datos pesados.
   - **Límite esperado**: 6h por job, recursos limitados (2 vCPU, 7GB RAM).
   - **Riesgo**: Ban de cuenta si se abusa para trading.
   - **Dificultad**: Media (YAML).
   - **Overnight**: No.
   - **Tests**: Sí (Pytest).
   - **Micro-probes**: No.
   - **Sweeps grandes**: No.
   - **Recomendación**: Solo para integridad y validación de código.
