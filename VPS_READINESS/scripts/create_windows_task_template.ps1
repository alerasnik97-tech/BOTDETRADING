
# PLANTILLA PARA CREAR TAREA PROGRAMADA EN WINDOWS
# Esta tarea arrancaría el preflight cada vez que se inicia la sesión o el sistema.

$action = New-ScheduledTaskAction -Execute 'PowerShell.exe' `
  -Argument '-ExecutionPolicy Bypass -File "C:\path\to\bottrading\VPS_READINESS\scripts\start_forward_demo_vps.ps1"'

$trigger = New-ScheduledTaskTrigger -AtLogOn

# NO EJECUTAR ESTO SIN AJUSTAR LA RUTA (path\to\bottrading)
# Register-ScheduledTask -Action $action -Trigger $trigger -TaskName "BotTrading_Forward_Demo" -Description "Arranque seguro del bot de trading en modo Demo"
