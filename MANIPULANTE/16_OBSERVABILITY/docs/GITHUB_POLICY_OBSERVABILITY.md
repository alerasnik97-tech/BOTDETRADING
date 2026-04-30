# MANIPULANTE Observability - GitHub Policy

## Subir a GitHub

Se puede subir:

- scripts Phase44 `.py`;
- docs `.md`;
- templates livianos;
- dashboard HTML si es liviano;
- BAT read-only para abrir dashboard;
- snapshots/resumenes `.md`;
- reportes finales Phase44;
- manifests de validacion livianos.

## No subir a GitHub

No subir:

- `MANIPULANTE/16_OBSERVABILITY/db/*.sqlite` si crece o si contiene datos sensibles;
- `MANIPULANTE/16_OBSERVABILITY/jsonl/*.jsonl` si crece;
- logs pesados;
- datos con tickets/cuenta si se consideran sensibles;
- secretos;
- `.env`;
- tokens;
- credenciales;
- configuraciones locales de proveedores;
- archivos MT5/MetaQuotes/Terminal;
- zips internos pesados.

## Reglas recomendadas de .gitignore

```gitignore
MANIPULANTE/16_OBSERVABILITY/db/*.sqlite
MANIPULANTE/16_OBSERVABILITY/jsonl/*.jsonl
MANIPULANTE/16_OBSERVABILITY/daily/*_daily_observability_report.json
```

Permitir:

```gitignore
!MANIPULANTE/16_OBSERVABILITY/docs/*.md
!MANIPULANTE/16_OBSERVABILITY/dashboard/*.html
!MANIPULANTE/16_OBSERVABILITY/dashboard/*.bat
!MANIPULANTE/16_OBSERVABILITY/daily/*.md
```

## Criterio practico

La observabilidad debe ser versionable en estructura y codigo, pero la evidencia operativa local debe quedarse local si puede revelar detalles de ejecucion o crecer demasiado.
