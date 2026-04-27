# EURUSD Stage 2 ChatGPT Handoff

Contenido:
- `EURUSD_MANUAL_ANNOTATION_STAGE2_REMAINDER.csv`: 55 trades restantes de ETAPA 2.
- `chartpacks/`: 55 chart packs PNG, uno por trade del CSV.
- `EURUSD_STAGE2_MASTER_INDEX.csv`: mapeo rank -> trade_id -> chart_filename -> chart_path, con side/outcome/time_block.
- `EURUSD_MANUAL_ANNOTATION_SCHEMA.md`: taxonomia cerrada para anotacion.

Uso recomendado:
1. Abrir `EURUSD_STAGE2_MASTER_INDEX.csv` para seguir el orden rank 1-55.
2. Revisar cada PNG en `chartpacks/`.
3. Anotar los 7 campos humanos sobre `EURUSD_MANUAL_ANNOTATION_STAGE2_REMAINDER.csv` usando el schema.