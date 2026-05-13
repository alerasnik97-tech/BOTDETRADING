# AUDITORÍA DE ESTABILIDAD MENSUAL — R1 CANDIDATE FACTORY

## 1. Análisis de Resiliencia de la Curva
El candidato `cfg_r1_factory_opt_001` presenta una distribución mensual madura:
- **Meses Positivos**: 50 (~66%)
- **Meses Negativos**: 26 (~34%)
- **Racha Negativa Máxima**: 3 meses.
- **Peor Mes**: -2.5 R (Octubre 2025).

## 2. Dependencia Transaccional
La auditoría de concentración confirma que el *edge* no reside en trades aislados. El PnL acumulado de las top 5 operaciones representa el `10.8%` del retorno neto total, lo cual indica una alta representatividad estadística del modelo.
