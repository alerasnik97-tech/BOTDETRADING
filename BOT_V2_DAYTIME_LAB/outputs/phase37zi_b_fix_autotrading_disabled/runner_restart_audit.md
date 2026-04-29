# PHASE37ZI-B RUNNER RESTART AUDIT

- timestamp: 2026-04-29T17:26:53
- runner_lock_removed: True
- start_process_stdout:
- start_process_stderr: Start-Process : No se puede validar el argumento del parÃƒÂ¡metro 'ArgumentList'. El argumento es null o estÃƒÂ¡ vacÃƒÂ­o.
Proporcione un argumento que no sea null o que no estÃƒÂ© vacÃƒÂ­o e intente ejecutar el comando de nuevo.
En lÃƒÂ­nea: 1 CarÃƒÂ¡cter: 54
+ Start-Process -FilePath "$env:ComSpec" -ArgumentList "/c", ""C:\Users ...
+                                                      ~~~~~~~~
    + CategoryInfo          : InvalidData: (:) [Start-Process], ParameterBindingValidationException
    + FullyQualifiedErrorId : ParameterArgumentValidationError,Microsoft.PowerShell.Commands.StartProcessCommand
- start_process_returncode: 1
- runner_pids_before_stop: [12512]
- runner_pids_after_stop: []
- runner_pids_after_start_wait: []

## Permisos despues de restart runner
- account_trade_allowed: True
- terminal_trade_allowed: False
- tradeapi_disabled: True
- positions_total: 0
