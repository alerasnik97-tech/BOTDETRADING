# AUDITORÍA DE SELECCIÓN TOP 5 V49

## 1. Verificación de Criterios
- **Seleccionado por TRAIN/VAL**: SÍ. Los finalistas fueron elegidos basándose únicamente en los datos de 2020-2024.
- **TEST no usado**: CONFIRMADO. No hay mención ni procesamiento de datos de 2025 en el ranking.
- **N Mínimo**: SÍ. Todos los finalistas tienen al menos 20 trades en la muestra agregada.
- **Estabilidad de Slippage**: SÍ. Se verificó que el PF no colapsa al aumentar el slippage de 0.2 a 0.3.
- **Max 3 trades/day**: CUMPLIDO. El motor impone este límite de forma nativa.
- **EOM Artificial**: NO. Se desactivó explícitamente cualquier métrica sintética de cierre de mes.

## 2. Perfil de los Finalistas
Los finalistas (Top 5) muestran una curva de equidad con tendencia positiva tanto en el periodo de entrenamiento como en el de validación, lo que sugiere una ventaja estadística real basada en la absorción en niveles clave.

## 3. Veredicto de Selección
**TOP 5 VALID = YES**
La selección es robusta y está lista para la validación final.
