# MASTER PROJECT INVENTORY

**Fecha de Auditoría:** 2026-04-27
**Estado Global:** CONSOLIDADO (Raíz Única)

## 1. Estructura de Directorios

| Carpeta | Estado | Archivos (aprox) | Propósito |
|---------|--------|------------------|-----------|
| `BOT_V2_DAYTIME_LAB` | ACTIVA | 276 | Laboratorio diurno Bot V2 (Research/Dev) |
| `STRATEGIES` | ACTIVA | 10 | Almacén de fases de estrategias validadas/rechazadas |
| `REPORTS` | ACTIVA | 162 | Reportes de auditoría, infraestructura y resultados |
| `DATA` | ACTIVA | 76 | Datasets consolidados (2015-2026) |
| `ARCHIVE_SUPERSEDED` | ARCHIVO | 351 | Versiones obsoletas, carpetas duplicadas y reportes viejos |
| `scripts` | ACTIVA | - | Motores de ejecución y utilidades |
| `scratch` | TEMPORAL | - | Scripts de un solo uso y pruebas rápidas |

## 2. Hallazgos de Auditoría

- **Raíz Oficial Única:** Confirmada en `C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo`.
- **ZIPs Duplicados:** Se detectaron dos ZIPs con el nombre `000_PARA_CHATGPT.zip`. Uno en la raíz (Maestro) y otro en `BOT_V2_DAYTIME_LAB` (Específico del laboratorio). El de la raíz es la autoridad máxima del proyecto.
- **Documentación de Autoridad:** Los documentos `00`, `01`, `02` y `03` están presentes y actualizados.
- **Jerarquía:** SCBI_M5_GLOBAL se mantiene como autoridad overnight protegida. Phase 7 y 8 son los únicos candidatos vivos en STRATEGIES.

## 3. Veredicto de Inventario
**ESTRUCTURA_VALIDADA.** El proyecto está libre de carpetas externas activas y mantiene una jerarquía documental clara.
