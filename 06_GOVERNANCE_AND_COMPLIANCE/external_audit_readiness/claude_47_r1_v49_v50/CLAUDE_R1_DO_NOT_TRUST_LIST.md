# LISTA INSTITUCIONAL DE EXCLUSIÓN FORENSE: "DO NOT TRUST"
**Estrategia:** R1 (Mean Reversion / NY Open Absorption)  
**Nivel de Severidad:** CRÍTICO (Bloqueo Contractual de Fuentes)  

---

Para preservar la higiene probatoria y evitar el resurgimiento de sesgos o métricas corrompidas, se ordena al agente externo Claude 4.7 High **IGNORAR, RECHAZAR Y DESCALIFICAR** de forma absoluta e irrevocable cualquiera de los siguientes elementos si son hallados durante su escrutinio:

## 1. Iteraciones Estructurales Invalidadas
- **Versión V43:** *DO NOT TRUST.* Resultados históricos severamente corrompidos por minería de datos y selección de hiperparámetros In-Sample sin resguardo Out-of-Sample válido.
- **Versión V44:** *DO NOT TRUST.* Simulaciones invalidadas por deficiencias graves en la captura de granularidad de ticks y latencia subestimada en la capa del motor.
- **Versión V46:** *DO NOT TRUST.* Salidas descartadas por dobles invocaciones y aplicación redundante de markdowns de slippage, distorsionando el factor de ganancia.

## 2. Artefactos y Métricas de Rendimiento Carentes de Pureza Física
- **Cualquier Reporte sin Bitácora de Trades:** Descalificar de forma inmediata todo resumen estadístico, pliego matricial o gráfica de rendimiento que no esté acompañado en la misma carpeta por su correspondiente bitácora transaccional granular (`*TRADES.csv`).
- **Cualquier Profit Factor que NO Recalcula desde Trades:** Rechazar incondicionalmente cualquier métrica de Profit Factor (PF), ganancia neta o Win Rate escalar declarada en texto que difiera de la suma matemática estricta calculada sobre las transacciones individuales del archivo físico.
- **Cualquier Resultado sin Auditoría de Conteo de Filas (Rowcount Audit):** Prohibido confiar en conjuntos de datos o salidas de simulación cuyo tamaño muestral $N$ no haya sido sometido y validado mediante un pre-vuelo de conteo físico automatizado.

## 3. Empaquetados y Evidencias Obsoletas
- **Cualquier Archivo ZIP Viejo o Secundario:** Descartar de forma absoluta todo empaquetado alternativo como `SUBIR_A_CHATGPT*`, `GATE*.zip` o copias de seguridad residuales. La única fuente canónica de empaquetado raíz es el archivo bloqueado por cerrojo único `000_PARA_CHATGPT.zip` regenerado de forma atómica.
