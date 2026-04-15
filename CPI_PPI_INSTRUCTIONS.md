# Instrucciones para Curacion Manual Asistida de CPI + PPI

## Estado: ESPERANDO DATOS MANUALES

## Que necesito de tu parte

Necesito que completes **72 fechas** (36 CPI + 36 PPI) usando el calendario oficial BLS.

### Opcion A: Formato Simple (Preferida)

Pasame las fechas en formato plano, ejemplo:

```
CPI 2024:
2024-01-11
2024-02-13
2024-03-12
...

PPI 2024:
2024-01-12
2024-02-15
2024-03-14
...
```

### Opcion B: Formato JSON

Edita directamente `CPI_PPI_manual_fill_template.json` y completame los campos `local_date_ny`.

### Fuentes oficiales a consultar

1. Abri tu navegador
2. And a: https://www.bls.gov/schedule/news_release/cpi.htm
3. Copia las fechas de release para 2024, 2025, 2026
4. Repeti para PPI: https://www.bls.gov/schedule/news_release/ppi.htm

### Horario confirmado

- **CPI**: 08:30 AM ET (America/New_York)
- **PPI**: 08:30 AM ET (America/New_York)

## Atencion especial 2025

El ano 2025 tuvo perturbaciones por "lapse in appropriations" (cierre de gobierno).
Algunos meses pueden tener fechas irregulares o releases cancelados.

Verifica especificamente:
- Octubre 2025: probablemente cancelado o combinado
- Noviembre 2025: posiblemente retrasado

## Que hare yo con los datos

1. Recibo tus fechas verificadas
2. Valido formato YYYY-MM-DD
3. Incorporo al manifest oficial: `data/official_anchors/manifests/user_curated_releases.json`
4. Ejecuto el pipeline
5. Genero reporte con cobertura actualizada
6. Actualizo el ZIP

## No tocar hasta confirmacion

El manifest oficial NO se modifica hasta recibir tus fechas verificadas.

## Archivos involucrados

- Plantilla: `CPI_PPI_manual_fill_template.json`
- Manifest final: `data/official_anchors/manifests/user_curated_releases.json`
- Reporte audit: `reports/official_anchors/`
- ZIP: `000_PARA_CHATGPT.zip`
