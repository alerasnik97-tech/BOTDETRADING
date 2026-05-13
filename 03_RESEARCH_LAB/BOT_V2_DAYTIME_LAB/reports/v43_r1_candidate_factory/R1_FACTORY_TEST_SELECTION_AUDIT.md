# AUDITORÍA DE SELECCIÓN DE TEST — R1 CANDIDATE FACTORY

## 1. Integridad del Proceso
Se certifica que la partición TEST (2025-2026) permaneció herméticamente cerrada durante toda la fase de escaneo y filtrado de la fábrica V43. 
- **Configs Escaneadas**: 1200
- **Filtro Top 20**: Basado en $PF_{val} \ge 1.15$ y robustez subperiodo.
- **Filtro Top 5 (Finalistas)**: Basado en estabilidad mensual y resiliencia al slippage 0.3 en validación.

## 2. Ejecución Única (Single-Run)
Los finalistas fueron ejecutados una sola vez sobre los datos de prueba. No se realizaron ajustes paramétricos ni se "re-seleccionaron" candidatos tras observar los resultados de TEST.

## 3. Veredicto de Candidato Líder
El candidato `cfg_r1_factory_opt_001` prevalece al sostener un $PF_{test} = 1.15$ neto, cumpliendo con la aserción de rentabilidad fuera de muestra exigida.
