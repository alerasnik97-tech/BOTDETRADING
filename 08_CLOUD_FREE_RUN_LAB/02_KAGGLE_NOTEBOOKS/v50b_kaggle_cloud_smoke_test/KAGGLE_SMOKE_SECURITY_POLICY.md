# KAGGLE_SMOKE_SECURITY_POLICY

- **No pegar tokens en notebook**: Queda terminantemente prohibido escribir o pegar Personal Access Tokens (PAT) directamente en las celdas del notebook.
- **No usar token Telegram**: No se permite el uso ni la configuración de bots de Telegram en esta fase.
- **No usar .env**: El archivo de variables de entorno no debe subirse ni crearse en el entorno de Kaggle.
- **No imprimir secrets**: Evitar cualquier sentencia `print` que pueda exponer información sensible.
- **Acceso a GitHub**: Si se requiere acceso a un repo privado, usar estrictamente Kaggle Secrets (`UserSecretsClient`), pero NUNCA imprimirlos. Para esta prueba se asume el uso de la branch pública/limpia si es posible.
- **Datos**: No subir datos privados sin confirmación explícita y auditoría.
- **Outputs**: Verificar que los archivos generados no contengan trazas de secretos o rutas locales sensibles.
- **Git Push**: No realizar `push` desde el entorno de Kaggle hacia GitHub en esta prueba.
