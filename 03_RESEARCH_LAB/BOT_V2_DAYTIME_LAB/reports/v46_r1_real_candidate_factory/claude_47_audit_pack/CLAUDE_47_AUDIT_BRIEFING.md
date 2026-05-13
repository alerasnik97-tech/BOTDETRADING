# CLAUDE 4.7 OPUS HIGH AUDIT BRIEFING — R1 V46 REAL FACTORY

## 1. Misión Forense
Se requiere una auditoría de máxima integridad sobre la Candidate Factory R1 V46. Tu objetivo es detectar cualquier rastro de artefactos sintéticos, alucinaciones métricas o fugas de información.

## 2. Puntos de Control Obligatorios
1. **Row Count Parity**: Verifica que `R1_V46_CANDIDATE_RANKING.csv` tenga exactamente 1201 filas para sustentar las 1200 configuraciones declaradas.
2. **N Match**: Verifica que el número de trades reportado coincida con las filas físicas en `R1_V46_TRADES.csv`.
3. **Blind Protocol**: Evalúa si hay indicios de que los candidatos del Top 5 fueron seleccionados con conocimiento previo de los resultados de TEST.
4. **Slippage Robustness**: Revisa `R1_V46_SLIPPAGE_STRESS.csv`. ¿El edge sobrevive a 0.3 pips netos?
5. **Causal Integrity**: Verifica si el motor V7 mantiene la causalidad absoluta tras la corrida.

## 3. Veredicto Final
Si encuentras una discrepancia superior al 0.1%, reporta BLOQUEO POR ARTEFACTO SINTÉTICO. Si la evidencia es sólida, sanciona el paso a la confirmación final.
