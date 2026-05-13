# GITHUB_ACTIONS_CI_USE_CASES

GitHub Actions sirve para:
- **pytest**: Ejecución automática de la suite de tests ante cambios.
- **lint**: Verificación de estilo y errores estáticos (flake8, black).
- **packaging validation**: Asegurar que el `CLOUD_PACKAGE` se puede construir sin errores.
- **testzip**: Validación automática de la integridad de los ZIPs de entrega.
- **hash checks**: Verificación de que los archivos críticos no han sido modificados accidentalmente.
- **smoke tests pequeños**: Corridas de 1-2 minutos para verificar que el motor arranca bien.
