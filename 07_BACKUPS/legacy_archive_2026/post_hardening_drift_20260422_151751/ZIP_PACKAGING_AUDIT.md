# ZIP Packaging Audit

Generated at: `2026-04-22T14:27:38.116972+00:00`

## Rebuild Status

- Reconstruccion desde cero: SI
- Reemplazo completo del zip anterior: SI
- Zip previo detectado: `339358` bytes
- Zips extra eliminados de la raiz: ninguno

## Canonical Content Criterion

- Se mantiene solo el set minimo vigente para entender el estado real del laboratorio y operar seguro dentro del proyecto.
- Se conserva la referencia al benchmark H6 solo como benchmark comparativo vigente.
- Se agregan solo artefactos canonicos reutilizables y resultados activos cuando existen fisicamente.
- `ZIP_DELIVERY_STATUS.md` se mantiene fuera del zip para evitar una autorreferencia imposible entre hash del zip y contenido interno.

## Exclusion Criterion Applied

- Se excluyen backups, archivos intermedios, staging y cualquier archivo con sufijo `_BACKUP_`.
- Se excluyen zips obsoletos de raiz distintos de `000_PARA_CHATGPT.zip`.
- Se excluyen scripts, codigo, datasets y resultados historicos que no son necesarios para entender el estado vigente y operar seguro dentro del proyecto.
- Se excluyen artefactos de etapas intermedias o handoffs ya superados.
- Se excluye todo archivo no nombrado explicitamente en la lista canonica.

## Integrity Checks

- Archivos canonicos incluidos: `94`
- Ausencia de duplicados logicos por nombre interno: SI
- Ausencia de backups dentro del zip: SI
- Ausencia de archivos intermedios dentro del zip: SI
- Coherencia con documentos canonicos activos: SI
- H6 preservado como benchmark vigente e intocable: SI
