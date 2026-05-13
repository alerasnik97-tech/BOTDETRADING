# ORACLE_SECURITY_CHECKLIST

- [ ] **No subir claves broker**: Verificado que el código no contiene credenciales hardcoded.
- [ ] **No subir tokens**: Verificado que no hay tokens de Telegram o APIs.
- [ ] **No abrir puertos innecesarios**: Solo puerto 22 (SSH) abierto en la VCN.
- [ ] **Usar SSH key**: No habilitar login por password.
- [ ] **No usar password débil**: Si se usa sudo password, que sea robusto.
- [ ] **No subir raw tick data completo**: Solo particiones necesarias para la corrida actual.
- [ ] **Borrar outputs temporales**: Si contienen datos sensibles, borrarlos tras la descarga.
- [ ] **Verificar costos en 0**: Revisar la sección de facturación para asegurar que sigue en Free Tier.
- [ ] **Revisar instancia**: Asegurar que la instancia no ha sido comprometida (revisar `last`, `history`).
