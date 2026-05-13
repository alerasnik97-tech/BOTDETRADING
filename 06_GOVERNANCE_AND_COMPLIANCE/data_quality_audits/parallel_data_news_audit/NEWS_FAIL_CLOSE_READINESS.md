# INSTITUTIONAL NEWS FAIL-CLOSE READINESS ASSESSMENT
**Target Consumer:** Research MANIPULANTE 3.0 Simulation Engine  
**Execution Type:** Read-Only Parallel Agent Forensic Output (Zero-Interference Lockdown Protocol)

---

### 1. Fuente Oficial de Calendario
La fuente institucional **OFICIAL** y autorizada para la orquestación de barridos de largo alcance en **Manipulante 3.0** es el archivo premium curado:  
`news_eurusd_am_fortress_v3.csv`

### 2. Cobertura Temporal Canónica
El repositorio oficial **AM Fortress v3** cuenta con registros validados de forma continua y determinística desde el **2 de enero de 2020** hasta el **30 de abril de 2026** (`2020-01-02` a `2026-04-30`), alineándose simétricamente con el límite superior físico de la serie de datos de mercado por ticks.

### 3. Protocolo de Transición para el Período 2015–2019
Dado que la serie premium AM Fortress v3 inicia su trazabilidad de alta fidelidad en enero de 2020, las particiones de entrenamiento de largo plazo que abarcan desde el **1 de enero de 2015 hasta el 31 de diciembre de 2019** deben enrutar sus lecturas hacia el archivo de curación estándar legacy:  
`news_eurusd_m15_validated.csv`  
Este archivo provee los anclajes de tiempo nativos suficientes para no romper la continuidad de los bucles causales, aunque con un nivel de granularidad de fuentes secundarias.

### 4. Viabilidad Operativa sin Calendario Premium
**NO**, el motor de **Research** tiene **ESTRICTAMENTE PROHIBIDO** operar o computar atribuciones causales en años recientes (2020 en adelante) prescindiendo del calendario premium. La capa de filtrado de noticias constituye un componente ortogonal crítico del edge de la estrategia; simular ejecuciones a ciegas en entornos de alta densidad macroeconómica introduce un sesgo de supervivencia letal y distorsiona el Profit Factor neto.

### 5. Clasificación de Particiones Mensuales
Para el consumo sin ambigüedad por parte de los scripts automatizados de CI/CD y pipelines de barrido, los meses quedan demarcados bajo los siguientes estados obligatorios:
- **`PASS` (Enero 2020 a Abril 2026):** Consumo directo y nativo desde `news_eurusd_am_fortress_v3.csv`. Totalmente auditados y alineados en zona horaria local de Nueva York.
- **`REVIEW` (Enero 2015 a Diciembre 2019):** Consumo habilitado desde `news_eurusd_m15_validated.csv`. Aprobado condicionalmente para fases tempranas de entrenamiento global, requiriendo revisión manual ante variaciones atípicas de volatilidad en los reportes de salida.
- **`FAIL_CLOSE` (Mayo 2026 en adelante o cualquier bloque faltante):** Interrupción absoluta de simulación.

### 6. Política de Bloqueo de Operaciones (Fail-Close Mechanism)
**SÍ**, la arquitectura del motor de **Manipulante 3.0** debe implementar una barrera dura de tipo **Fail-Close**. Si durante el avance del bucle de simulación por eventos el gestor de contexto detecta la ausencia física del archivo de calendario correspondiente o una brecha de cobertura de registros superior a 5 días hábiles, el sistema debe **bloquear inmediatamente toda emisión de órdenes (tanto entradas como cierres gestionados)** y arrojar una excepción fatal en terminal para evitar la contaminación silenciosa de los reportes de backtest.

### 7. Recomendación Final para el Gate 3.0
Se recomienda **APROBAR** el pase de compuerta hacia la orquestación de barridos de Manipulante 3.0 condicionada a la implementación estricta de un **Gestor de Enrutamiento Híbrido** en la inicialización del `Runner`. Dicho gestor debe cargar dinámicamente el baseline legacy para las fechas $t < \text{2020-01-01}$ y conmutar de forma transparente al vector premium AM Fortress v3 para $t \ge \text{2020-01-01}$, manteniendo activa la aserción de Fail-Close como capa de seguridad perimetral.
