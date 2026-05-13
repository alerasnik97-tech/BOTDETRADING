# CLOUD RUN - README [TEMPLATE]

## Proyecto: [Nombre de la Corrida]
## Fecha: [Fecha]
## Operador: [Agente/Usuario]

### Instrucciones
1. Activar venv: `source venv/bin/activate`
2. Verificar hashes: `sha256sum -c manifest.csv`
3. Ejecutar: `python runner.py --config config_cloud.json`
4. Monitorear logs: `tail -f outputs/run.log`

### Notas
- Corrida nocturna estimada: [X] horas.
- Límite de stop: [PF < 1.0].
