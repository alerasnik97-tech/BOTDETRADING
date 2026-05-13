# CLOUD_PACKAGE_BUILD_POLICY

- **No crear ZIP dentro del proyecto**: Los archivos del paquete deben estar sueltos en su carpeta de staging o fuera del proyecto.
- **Ubicación Externa**: Si se necesita empaquetar en un solo archivo para subir, hacerlo fuera del proyecto en: `C:\Users\alera\Desktop\CLOUD_UPLOAD_PACKAGES\`.
- **Fuente de Verdad**: El paquete cloud nunca es la fuente de verdad. Si el código cambia localmente, el paquete debe reconstruirse.
- **Auditoría de Salida**: Los outputs vuelven al proyecto solo después de una auditoría de integridad y formato.
- **Revalidación Local**: Todo resultado exitoso en la nube debe revalidarse localmente antes de ser aceptado como "Edge" válido.
