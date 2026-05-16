# ROCKI AM - PLAN DE IMPLEMENTACION VPS

Para retomar esta estrategia en el futuro, se deben seguir los siguientes pasos técnicos y operativos:

## 1. Fase de Re-Activacion (Investigacion)
- [ ] Ejecutar `05_CODIGO_REFERENCIA/ROCKI_AM_validation_runner.py` para asegurar que el entorno de Python sigue siendo compatible.
- [ ] Realizar una auditoría de costos netos (Phase 38B style) para confirmar la rentabilidad tras comisiones de Prop Firm reales.
- [ ] Verificar el impacto de noticias nocturnas (London Open) sobre los barridos de liquidez.

## 2. Fase de Infraestructura (VPS)
- [ ] Contratar un VPS con baja latencia hacia los servidores de MT5 (London/New York).
- [ ] Configurar una instancia de MT5 independiente para **ROCKI AM**.
- [ ] Asegurar redundancia de conexión y suministro eléctrico.

## 3. Fase de Desarrollo (Runner)
- [ ] Adaptar el runner de MANIPULANTE para la lógica SCBI M5.
- [ ] Implementar un `STATUS_ROCKI_AM.bat` que funcione en horario nocturno.
- [ ] Configurar las alertas de Telegram/Discord para monitoreo remoto desde el móvil durante la madrugada.

## 4. Fase de Certificacion (Demo)
- [ ] Operar 2 semanas en Demo con el runner automático.
- [ ] Comparar `exit_price` real vs `exit_price` teórico para medir el slippage nocturno.

## 5. Fase de Lanzamiento
- [ ] Iniciar en una cuenta de fondeo pequeña (5k o 10k) o como diversificación de la cuenta principal.

---
*Este plan garantiza que el rescate de hoy se convierta en una operacion profesional mañana.*
