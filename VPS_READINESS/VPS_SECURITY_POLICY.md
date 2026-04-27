# VPS SECURITY POLICY

Esta política es de cumplimiento obligatorio para todo operador del laboratorio.

## 1. Gestión de Credenciales
- **PROHIBIDO** versionar `mt5_local_config.json`.
- **PROHIBIDO** subir archivos `.env` a GitHub.
- Los passwords deben ser introducidos manualmente en la VPS y no almacenados en texto claro fuera de la configuración local protegida.

## 2. Restricciones de Cuenta
- En la etapa actual, **SOLO se permiten cuentas DEMO**.
- La conexión a una cuenta REAL en la VPS resultará en la suspensión inmediata del entorno de ejecución.

## 3. Integridad de Código
- No instalar paquetes de Python externos a `requirements.txt` sin previa validación en local.
- No ejecutar scripts de origen desconocido.
- Todas las actualizaciones de código deben llegar a través de la rama segura `chore/github-clean-sync`.

## 4. Ejecución Segura
- El bot debe operar siempre con `allow_live=false`.
- El News Guard es obligatorio. Operar sin protección ante noticias de alto impacto invalida la sesión.
- Cada trade debe tener un SL (Stop Loss) enviado al servidor de MT5 al momento de la apertura.
