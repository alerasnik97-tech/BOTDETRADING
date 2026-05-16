# INFRASTRUCTURE CURRENT TRUTH

**Fecha de Auditoría:** 2026-04-27
**Veredicto Global:** INFRA_OK (Con advertencias operativas)

## 1. Estado de Componentes Críticos

### 1.1 BID/ASK/SPREAD
- **Estado:** REPARADO Y CERTIFICADO.
- **Detalle:** Se migraron los datasets 2020-2026 de BID-only a BID/ASK/SPREAD real.
- **Motores:** Bot V2 usa el pipeline de precisión máxima. Motores obsoletos en `ARCHIVE_SUPERSEDED` pueden contener trazas de BID-only.

### 1.2 News Guard
- **Estado:** VALIDADO PARA RESEARCH.
- **Detalle:** El News Guard está integrado en el motor Bot V2.
- **Pendiente:** Verificación de latencia antes de cualquier pase a live.

### 1.3 Python Runtime
- **Estado:** OPERATIVO.
- **Detalle:** Pandas, Numpy y Scipy importan correctamente. No se detectan bloqueos de DLL en el entorno actual.

### 1.4 Horarios Operativos
- **SCBI_M5_GLOBAL:** Madrugada/Londres (Overnight). Es el edge principal.
- **Bot V2:** Diurno independiente. 
- **Nota:** El rango 07:00–20:30 es disponibilidad operativa, no garantiza edge automático sin filtros.

### 1.5 Sunday Gap / Loader
- **Estado:** REPARADO.
- **Detalle:** Los loaders actuales filtran o gestionan correctamente el gap del domingo. No usar loaders anteriores a la auditoría de abril 2026.

### 1.6 Bug 19:00
- **Estado:** CONTROLADO.
- **Detalle:** Solo afecta a pruebas de estrés con configuraciones horarias específicas ya identificadas. No hay riesgo en las estrategias Phase 7/8 bajo sus parámetros actuales.

## 2. Veredicto de Infraestructura
```
╔════════════════════════════════════════════════════════════╗
║  INFRA_OK                                                  ║
║  Datasets certificados. Motores validados.                 ║
║  Prohibido usar componentes de ARCHIVE_SUPERSEDED.         ║
╚════════════════════════════════════════════════════════════╝
```
