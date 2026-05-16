# VPS TROUBLESHOOTING GUIDE

Soluciones rápidas a problemas comunes en el despliegue en VPS.

## 1. Problemas de Python / Pip
- **"python" no reconocido:** Asegúrate de marcar "Add Python to PATH" durante la instalación o añade la ruta manualmente a las variables de entorno.
- **Error en la instalación de MetaTrader5:** Requiere Python de 64 bits. Verifica con `python -c "import struct; print(struct.calcsize('P') * 8)"`.

## 2. Problemas de MetaTrader 5
- **"initialize() falló":** Verifica que la ruta `mt5_terminal_path` en `mt5_local_config.json` sea correcta (usa doble barra invertida `\\`).
- **"Fallo al loguear":** Revisa que el login, password y servidor coincidan exactamente con los de tu cuenta demo.
- **Símbolo EURUSD no encontrado:** Asegúrate de que el símbolo esté habilitado en el Market Watch de MT5.

## 3. Problemas de Seguridad / Bloqueos
- **"¡CUENTA REAL DETECTADA!":** El script ha bloqueado la conexión porque los datos de cuenta indican que es real. Cierra MT5, cambia a una cuenta demo y reinicia.
- **"Ruta obsoleta detectada":** Estás ejecutando código viejo. Realiza un `git pull` de la rama `chore/github-clean-sync`.

## 4. Problemas de Logs
- **"Permission Denied":** La VPS no tiene permisos para escribir en la carpeta `logs`. Ejecuta PowerShell como Administrador.
