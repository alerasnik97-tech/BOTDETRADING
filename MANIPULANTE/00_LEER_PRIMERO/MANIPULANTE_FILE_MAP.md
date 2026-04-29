# MANIPULANTE - MAPA DE ARCHIVOS (FILE MAP)

Este documento describe la estructura de la carpeta oficial del bot **MANIPULANTE**.

## 1. Raiz Operativa (Uso Diario)
En la raiz solo deben residir los botones de ejecucion principal:
- **START_MANIPULANTE.bat**: Inicia el bot y verifica duplicados.
- **STATUS_MANIPULANTE.bat**: Muestra el panel de control actualizado cada 30s.
- **STOP_MANIPULANTE.bat**: Detiene el bot de forma segura (valida posiciones abiertas).

## 2. Carpetas de Documentacion y Operacion

| Carpeta | Descripcion | Uso |
| :--- | :--- | :--- |
| **00_LEER_PRIMERO** | Documentos maestros, reglas y manuales operativos. | **LECTURA OBLIGATORIA** |
| **01_ESTRATEGIA_AUTORIDAD** | Parametros oficiales y definicion logica de la estrategia. | Referencia tecnica |
| **02_REGLAS_DE_FONDEO** | Limites y parametros especificos para cuentas Prop Firm. | Referencia tecnica |
| **03_MT5_DEMO_LAUNCHER** | Accesos directos y configuracion para lanzar terminales demo. | Operativo |
| **04_OPERACION_DIARIA** | Archivos de control y estados temporales del bot. | Sistema (No tocar) |
| **05_DUAL_LEDGER_SHADOW** | Registro para comparar operaciones reales vs simuladas. | Auditoria |
| **06_TEMPLATES** | Plantillas para la generacion de reportes y auditorias. | Sistema |
| **07_REPORTES_CLAVE** | Historial de rendimiento y cierres mensuales. | Auditoria |
| **08_CHECKLISTS** | Pasos a seguir antes de iniciar la sesion de trading. | Operativo |
| **09_COMPLIANCE** | Validacion de cumplimiento de reglas de gestion de riesgo. | Auditoria |
| **10_LOGS_PAPER** | Archivos de log generados durante la ejecucion en demo. | Auditoria |
| **11_GITHUB_SYNC_NOTES** | Documentacion de versiones y cambios en el codigo. | Auditoria |
| **12_MICRO_REAL_READINESS** | Reportes de aptitud para el paso a cuenta real. | Auditoria |
| **13_FTMO_TRIAL_AUTOMATION** | Logica interna de la automatizacion para FTMO Trial. | Sistema (No tocar) |
| **14_ANALISIS** | Auditorias profundas de costos (Phase 38B) y modelos. | Auditoria |
| **99_ARCHIVO_BAT_ANTIGUOS** | Archivos .bat obsoletos o de uso tecnico avanzado. | Archivo |

## 3. Archivos Criticos (NO TOCAR)
- Cualquier archivo dentro de `13_FTMO_TRIAL_AUTOMATION` (el bot runner).
- Archivos `.json` de configuracion en `01_ESTRATEGIA_AUTORIDAD`.
- Rutas de instalacion de Python.

## 4. Trazabilidad
**MANIPULANTE** es el bot oficial, consolidado a partir de las auditorias de la **Phase 25**. Toda la documentacion historica bajo el nombre "Phase 25" se considera evidencia fundacional de la robustez de este bot.
