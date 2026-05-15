# V50B LOCK FILE POLICY

Todo proceso de ejecución masiva DEBE adquirir un lock file antes de procesar el primer bar. Si el lock está stale (proceso muerto), requiere borrado manual y registro en logs de auditoría.
