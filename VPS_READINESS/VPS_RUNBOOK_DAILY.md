# VPS DAILY RUNBOOK

Instrucciones diarias para la operación del laboratorio en VPS.

## 1. Inicio de Jornada (Checklist)
- [ ] VPS Online (RDP Conectado).
- [ ] MT5 Abierto y conectado a cuenta **DEMO**.
- [ ] Reloj del sistema sincronizado.
- [ ] Ejecutar `vps_preflight_check.ps1`.
- [ ] Verificar `allow_live=false` en el log de inicio.
- [ ] Confirmar que el News Guard tiene el calendario actualizado.

## 2. Durante la Sesión
- [ ] Monitorear el archivo de Heartbeat (latido del sistema).
- [ ] Revisar el log de operaciones si ocurre algún trade.
- [ ] **NO INTERVENIR** manualmente en las operaciones abiertas por el bot.
- [ ] Registrar cualquier desconexión o latencia alta.

## 3. Cierre de Jornada
- [ ] Exportar reporte diario de operaciones.
- [ ] Realizar backup de los archivos de logs si es necesario.
- [ ] Verificar que no se hayan creado archivos basura en la raíz.
- [ ] Documentar el veredicto del día en el diario de forward testing.
