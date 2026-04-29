# PLAN DE CONEXIÓN A CUENTA MICRO REAL

Este documento detalla los pasos obligatorios para conectar MANIPULANTE a una cuenta real pequeña por primera vez.

## Fase 1: Preparación MT5
1. Usar siempre el launcher `ABRIR_MANIPULANTE_DEMO.bat` (aunque sea para real, para asegurar modo portable y visibilidad de logs).
2. Verificar que el AutoTrading esté **DESACTIVADO** en la barra superior.
3. Asegurarse de que no haya EAs cargados en los gráficos que tengan permiso de trading.

## Fase 2: Conexión de Cuenta
1. Conectar manualmente la cuenta real.
2. Verificar el balance.
3. Verificar que el símbolo EURUSD esté disponible y tenga un spread razonable (< 1.5 pips).
4. Verificar el `Lot Step` (debe ser 0.01).

## Fase 3: Protocolo del Primer Trade
1. **Riesgo Máximo Inicial: 0.10% a 0.25%.**
2. NO usar 0.75% ni 1.00% bajo ninguna circunstancia en la primera semana.
3. El trade debe ser ejecutado **MANUALMENTE** siguiendo las señales del sistema.
4. Antes de abrir, usar `BOT_V2_DAYTIME_LAB/src/phase35_lot_size_validator.py` para confirmar el lotaje.
5. SL y TP deben ponerse **EN EL MOMENTO DE LA APERTURA**.

## Fase 4: Kill-Switch Real
Si algo falla (lotaje incorrecto, retraso, spread alto), cerrar inmediatamente y pausar por 24h.
