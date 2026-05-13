# POLÍTICA DE BLOQUEO DEFINITIVO DEL MOTOR CENTRAL (ENGINE CORE LOCKDOWN POLICY)

## 1. Declaración de Principios
El motor cuantitativo central (componentes V6 y V7) es la **Fuente de Verdad Inmutable** sobre la cual se asienta la validez causal, temporal, contable y operacional de toda investigación dentro de Antigravity/Bot de Trading. 
Para garantizar que ninguna optimización de estrategia corrompa las métricas mediante la relajación de costos institucionales o introducción de sesgos de look-ahead, se establece el presente **Régimen de Bloqueo Estricto**.

## 2. Alcance (Core Intocable)
Quedan bajo protección absoluta de inmutabilidad los siguientes directorios y todos sus sub-elementos:
- `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/src/v6_utils/`
- `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/src/v7_engine/`

## 3. Regla de Oro para el Desarrollo de Estrategias
**"Las estrategias se adaptan al motor; el motor jamás se adapta a la estrategia."**
Toda nueva hipótesis (R1, M4, etc.) debe consumir el motor central exclusivamente a través de sus interfaces públicas y firmas canónicas. Si una estrategia requiere funcionalidades de pre-procesamiento o estructuras de datos personalizadas, se deberán implementar **adaptadores (adapters) o envoltorios (wrappers)** en el espacio de nombres de la estrategia (ej. `src/R1/` o `src/strategies/`), preservando intacto el código del core.

## 4. Mecanismos de Enforcement
La inmutabilidad del core no depende únicamente de la disciplina humana, sino que está respaldada por:
1. **Manifiesto Criptográfico**: Registro continuo de hashes SHA256 de todos los archivos fuente (`ENGINE_CORE_HASH_MANIFEST.json`).
2. **Auditoría Continua**: Ejecución del script `ENGINE_CORE_VERIFY.py` que bloquea la ejecución de pruebas walk-forward o empaquetados en la nube ante la menor divergencia (drift), adición o eliminación de archivos en las rutas protegidas.
3. **Barrera de Repositorio (Pre-Commit Hook)**: Bloqueo a nivel de Git que impide registrar confirmaciones (commits) sobre archivos del core a menos que exista un Change Request explícitamente aprobado por la Dirección Quant.
