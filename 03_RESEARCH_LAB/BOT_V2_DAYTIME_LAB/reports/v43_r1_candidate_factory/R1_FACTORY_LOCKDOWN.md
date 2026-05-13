# MANIFIESTO DE BLOQUEO ARQUITECTÓNICO — R1 CANDIDATE FACTORY

## 1. Perímetro de Inmutabilidad Funcional (Core Lockdown)
Se sanciona formalmente la apertura de la fase **V43** (*Candidate Factory*), quedando bajo un candado hermético de no-modificación sobre los activos centrales del sistema:
- **Cero Deriva en el Core**: Se prohíbe de forma sagrada e incondicional alterar, parchar o reescribir un solo byte de los módulos ubicados en `src/v7_engine` y `src/v6_utils`. 
- **Fidelidad Estructural**: No se inyectarán nuevos activos de negociación o indicadores ajenos a la familia R1. La fábrica opera sobre el motor V7 certificado.

## 2. Restricciones Físicas de la Corrida
- **Instrumento Único**: `EURUSD`.
- **Ventana de Sesión**: Confinada estrictamente entre las `07:00` y las `17:00` NY (Foco 08-11).
- **Límite de Frecuencia**: Máximo 3 operaciones diarias por configuración.
- **Deducciones Obligatorias**: Slippage de 0.2 pips oficial + comisiones FTMO aplicadas bit-a-bit.

## 3. Veto de Selección por TEST
Se prohíbe el uso de la partición 2025-2026 para fines de optimización, filtrado o ranking. Cualquier violación a este precepto anulará la validez de los candidatos.
