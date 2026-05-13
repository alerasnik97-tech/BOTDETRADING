# KAGGLE_SECURITY_CHECKLIST

- [ ] **Secret GH_TOKEN**: Verificado que el token se obtiene de `UserSecretsClient` y no está en el código.
- [ ] **Limpieza de .netrc**: Celda de borrado de credenciales presente y ejecutada.
- [ ] **No print(token)**: No existen sentencias print que puedan exponer el secreto.
- [ ] **Visibilidad del Notebook**: El notebook de Kaggle debe ser "Private" si contiene acceso a código propietario.
- [ ] **No Datos Sensibles**: No se han subido archivos de configuración con claves de brokers o tokens de Telegram.
- [ ] **Dataset Privado**: Si se usa dataset en lugar de git, verificar que sea privado.
