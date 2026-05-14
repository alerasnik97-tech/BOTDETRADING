# V50B ANTI-PLACEHOLDER RULES

Para evitar el autoengaİo tǸcnico, se establecen las siguientes reglas inviolables:

1. **Evidencia Fsica Real**: Prohibido el uso de `np.random` o cualquier generador aleatorio para mǸtricas de performance (PF, WR, N, Total_R).
2. **Trazabilidad de Timestamps**: Todo trade debe tener un `entry_time` y `exit_time` real que corresponda a un dato de tick/bar de la ventana TRAIN/VAL autorizada.
3. **Validacin del Motor**: Los resultados solo se consideran vǭlidos si provienen de la salida directa del `UnifiedV7Engine`.
4. **Auditora de Cadenas Prohibidas**: Todo informe de resultados debe ser escaneado en busca de trminos como: *synthetic, dummy, fake, placeholder, random, simulated*.
5. **No Pass Sin Trades Reales**: Ninguna familia puede ser declarada APPROVED sin un archivo de trades con precios de ejecucin reales (Bid/Ask) del Vault.
6. **Alertas de Inconsistencia**: Si el recalculo de mǸtricas detecta timestamps fijos o uniformes (ej: todos en Mayo 2022), la fase queda automáticamente BLOQUEADA por fraude de evidencia.
