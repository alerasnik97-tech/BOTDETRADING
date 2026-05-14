# R1 V49.7B-R2B ?" FAILURE SUMMARY

**Estado**: CR?TICO. La familia R1 no muestra edge robusto bajo condiciones de costos institucionales y representatividad temporal.

## MǸtricas Clave
- **Total Configs**: 800
- **Configs PF_train >= 1.0**: 6 (0.75%)
- **Configs PF_val >= 1.15**: 27 (3.3%)
- **Configs passing BOTH**: **0 (0.0%)**

## Causas Principales del Fracaso
1. **Debilidad Estructural en TRAIN**: La mediana de PF en TRAIN es de **0.65**. El modelo pierde dinero de forma sistemǭtica en 2020, 2021 y 2022.
2. **Concentracin Temporal Extrema**: Los candidatos que "brillan" en VAL dependen en mǭs de un 80% de un slo mes (ej. Enero 2023 o Octubre 2024). Sin ese mes especfico, el PF_val cae por debajo de 1.0.
3. **Divergencia IS/OOS**: No hay correlacin positiva entre un buen resultado en TRAIN y un buen resultado en VAL. Esto sugiere que los resultados positivos son mayoritariamente **ruido estadstico**.
4. **Costos y Slippage**: El modelo R1 genera muchos trades con un edge por trade muy pequeño, lo que lo hace extremadamente sensible al spread y comisin.

**Veredicto Preliminar**: La familia R1, basada en absorcin de niveles diarios, no tiene esperanza de ser rentable bajo el runner actual.
