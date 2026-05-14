# KAGGLE_V49_7C_README - Full Scope Runner Readiness

Este documento prepara el entorno para la ejecución de la fase V49.7C en Kaggle.

## Objetivo
Ejecutar el sweep completo de V49.7C (600-1200 configs) en la nube para liberar la PC local, manteniendo la integridad total del proyecto y la seguridad de los secretos.

## Fuente de Verdad
- **GitHub**: alerasnik97-tech/bottrading
- **Branch**: clean-sync-branch

## Flujo de Trabajo
1. Preparar el Notebook de Kaggle con las celdas provistas.
2. Configurar el `GH_TOKEN` en los Secrets de Kaggle.
3. Clonar el repositorio y verificar el branch.
4. Ejecutar el sweep bajo supervisión periódica.
5. Devolver los resultados vía GitHub o descarga manual según la política de handoff.
