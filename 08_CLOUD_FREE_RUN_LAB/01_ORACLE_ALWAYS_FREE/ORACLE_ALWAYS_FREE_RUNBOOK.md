# ORACLE_ALWAYS_FREE_RUNBOOK

## Paso 1: Aprovisionamiento
- Acceder a la consola de Oracle Cloud.
- Crear Instancia de Cómputo.
- Seleccionar "Always Free Eligible".
- Preferencia: VM.Standard.A1.Flex (ARM) o VM.Standard.E2.1.Micro (AMD).
- Imagen: Ubuntu 22.04 LTS o 24.04 LTS.
- Descargar y guardar la SSH Key privada.

## Paso 2: Configuración del Sistema
- Acceder vía SSH: `ssh -i key.key ubuntu@<IP>`.
- Actualizar: `sudo apt update && sudo apt upgrade -y`.
- Instalar dependencias: `sudo apt install python3-pip python3-venv tmux -y`.

## Paso 3: Despliegue de Código
- Crear venv: `python3 -m venv venv`.
- Activar: `source venv/bin/activate`.
- Copiar solo el `CLOUD_PACKAGE` liviano.
- Instalar requirements: `pip install -r requirements.txt`.

## Paso 4: Ejecución
- Iniciar sesión de tmux: `tmux new -s trading`.
- Ejecutar runner: `python runner.py --config config.json`.
- Desacoplar: `Ctrl+B` y luego `D`.

## Paso 5: Recuperación de Resultados
- Verificar logs periódicamente.
- Al finalizar, comprimir outputs y descargar vía SCP/SFTP.
- Limpiar carpeta de outputs en la instancia.

## Notas
- Nunca guardar secrets.
- Nunca conectar broker.
- Nunca correr sin límites.
