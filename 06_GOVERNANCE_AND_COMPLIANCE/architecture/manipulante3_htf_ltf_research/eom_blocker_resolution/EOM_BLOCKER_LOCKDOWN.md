# EOM BLOCKER LOCKDOWN

Status: ACTIVE

Scope permitido:
- Auditar y corregir el tratamiento de EOM artificial en MANIPULANTE 3.0.
- Reejecutar solo las configs precomprometidas: CFG_002, CFG_005, CFG_004, CFG_001, CFG_003.
- Integrar los mismos constraints Data/News de Maximum Confirmation.
- Excluir EOM artificial de metricas principales.
- Documentar decision sin optimizar ni salvar la estrategia.

Prohibiciones confirmadas:
- No sweep grande.
- No overnight.
- No optimizacion.
- No nuevas combinaciones.
- No seleccion por TEST.
- No cambios en produccion.
- No cambios en incubacion.
- No mutacion de datos ni parquet.
- No push.
- No Explorer.
- No declaracion de listo para demo, fondeo o live.

Principio operativo:
Mientras un cierre EOM no pueda demostrarse como cierre real de sesion o fin real de datos, se trata fail-closed y no entra en metricas.

