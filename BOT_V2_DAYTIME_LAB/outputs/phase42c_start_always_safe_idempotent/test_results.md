# Phase42C Start Idempotence Tests

- total: 8
- pass: 8
- fail: 0

## Results
- PASS: START con STOP_BOT activo y sin posicion permite limpiar e iniciar
- PASS: START con bot activo no duplica
- PASS: START tocado 3 veces conserva un solo permiso de arranque
- PASS: STATUS con STOP_BOT activo muestra BOT DETENIDO
- PASS: START con cuenta real simulada hace emergency abort
- PASS: START con Exness simulado hace emergency abort
- PASS: START con posicion abierta no limpia ni reinicia
- PASS: Seguridad estatica de START
