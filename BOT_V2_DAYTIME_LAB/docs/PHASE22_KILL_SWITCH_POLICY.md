
# PHASE 22 KILL SWITCH POLICY (FAIL-CLOSED)

## 1. CONDICIONES DE ACTIVACIÓN (DEMO)
El proceso de Forward Demo debe detenerse inmediatamente si:
- **Hash Mismatch**: La configuración auditada ha sido alterada.
- **Safety Violation**: Se ejecuta un trade durante una ventana bloqueada por News o Data Mask.
- **Technical Failure**: El feed de datos falla por > 15 minutos en sesión.
- **Management Error**: Se detecta un trade sin SL o TP asignado.
- **User Intervention**: El usuario modifica el SL/TP de un trade activo en la terminal.

## 2. PROCEDIMIENTO
1. Cerrar todas las posiciones abiertas en la cuenta Demo.
2. Inactivar el bot/script de entrada.
3. Registrar la causa del fallo en `outputs/phase23_consistency_repair/failures/`.
4. La demo no se reiniciará hasta una nueva auditoría forense del fallo.
