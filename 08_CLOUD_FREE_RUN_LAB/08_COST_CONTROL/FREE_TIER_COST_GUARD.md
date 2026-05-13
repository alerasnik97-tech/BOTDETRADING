# FREE_TIER_COST_GUARD

- **Revisar Siempre**: Verificar manualmente que los recursos creados están dentro del "Always Free Tier".
- **No Activar Recursos Pagos**: Prohibido el uso de instancias con GPU de pago, almacenamiento premium o ancho de banda excesivo.
- **Limpieza de Discos**: No dejar volúmenes de disco (Boot Volumes) huérfanos; Oracle cobra por almacenamiento tras cierto límite.
- **Control de IPs**: Liberar IPs públicas si no se están usando.
- **Alertas**: Configurar alertas de presupuesto (Budget Alerts) en el proveedor (Oracle/Google) con un umbral de 0.01 USD.
- **No Cargar Tarjeta para Gastar**: Si el proveedor requiere tarjeta, que sea solo para verificación; nunca autorizar cargos automáticos para escalar recursos.
- **Alerta al Usuario**: El bot o el agente debe alertar al usuario antes de cualquier paso que pueda tener un costo potencial.
