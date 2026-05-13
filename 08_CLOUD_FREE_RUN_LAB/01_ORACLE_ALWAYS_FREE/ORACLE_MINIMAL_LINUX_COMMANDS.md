# ORACLE_MINIMAL_LINUX_COMMANDS

## SSH & Transferencia
- `ssh -i key.pem ubuntu@IP`: Conectar.
- `scp -i key.pem -r local_dir ubuntu@IP:~/remote_dir`: Subir archivos.
- `scp -i key.pem ubuntu@IP:~/remote_file local_dir`: Descargar archivos.

## Sistema
- `htop`: Ver consumo de CPU/RAM.
- `df -h`: Ver espacio en disco.
- `free -m`: Ver RAM libre.
- `tail -f logs.txt`: Seguir logs en tiempo real.

## Tmux (Persistencia)
- `tmux`: Iniciar sesión.
- `tmux attach`: Reconectar a la última sesión.
- `tmux ls`: Listar sesiones.
- `Ctrl+B, D`: Desacoplar sesión (dejar corriendo en segundo plano).

## Python
- `python3 -m venv venv`: Crear entorno virtual.
- `source venv/bin/activate`: Activar.
- `pip install -r requirements.txt`: Instalar dependencias.
