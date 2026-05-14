# R1 V49.7B-R2B ?" HYPOTHESIS DECISION

## Evaluacin de Hiptesis

### H4: R1 estǭ muerto como familia y debe congelarse.
- **Evidencia a favor**: 0/800 pasan el gate. Mediana de PF < 1.0 en todas las dimensiones. Alta concentracin temporal.
- **Evidencia en contra**: Algunos candidatos aislados en VAL con PF > 1.5.
- **Confianza**: ALTA.
- **Accin**: CONGELAR.

### H2: R1 tiene edge solo en VAL y es probable overfit.
- **Evidencia a favor**: Los resultados en VAL son sistemǭticamente mejores que en TRAIN (0.8 vs 0.65) pero siguen siendo perdedores en mediana. Los "ganadores" estǭn hiper-concentrados en meses de alta volatilidad.
- **Confianza**: ALTA.

### H1: R1 tiene edge real pero la muestra representativa castig TRAIN.
- **Evidencia a favor**: Ninguna clara. 2020-2022 cubren escenarios variados y el rendimiento es pobre en todos.
- **Confianza**: MUY BAJA.

## Conclusin de Investigacin
La familia R1 (Absorption Day-Time) no presenta un edge explotable. Continuar invirtiendo recursos en V49.7C (Full Scope) de esta familia es estadsticamente ineficiente.

**Recomendacin**: Mover R1 a "Research Archive" y pivotar hacia una nueva familia de estrategias (ej. R2 o V6-based).
