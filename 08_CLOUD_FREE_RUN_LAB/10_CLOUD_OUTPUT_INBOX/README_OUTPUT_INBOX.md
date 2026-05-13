# README_OUTPUT_INBOX

Este es el lugar donde se deben colocar los outputs descargados de la nube antes de ser procesados o auditados.

## Reglas de la Bandeja de Entrada
- Los archivos aquí colocados se consideran **NO VÁLIDOS** hasta que pasen la auditoría de integridad.
- Cada entrega debe venir en su propia carpeta con nombre descriptivo: `YYYYMMDD_STRATEGY_PROVIDER`.
- Se debe incluir:
  - `manifest.csv` con hashes de los outputs.
  - `config.json` usado en la corrida.
  - `runner_hash.txt` (SHA256 del código que lo ejecutó).
  - `run.log` completo.
- Si falta alguno de estos elementos, el estado del resultado pasa a **BLOCKED** y no se integra al laboratorio de investigación.
