# MT5 Demo Executor Lab - Guía de Operación

## 1. Requisitos Previos
- MetaTrader 5 instalado con una **Cuenta Demo** activa.
- Python 3.10+ con las siguientes librerías:
  - `MetaTrader5`
  - `pandas`
  - `pytz`

## 2. Configuración
1. Abre tu terminal MetaTrader 5 y asegúrate de estar logueado en la cuenta demo.
2. En MT5, ve a `Herramientas` -> `Opciones` -> `Asesores Expertos` y marca:
   - `Permitir el comercio algorítmico`
   - `Permitir la importación de DLL`
   - `Permitir WebRequest para las URL listadas` (si fuera necesario).

## 3. Ejecución
Para iniciar el ejecutor en modo sandbox, corre el siguiente comando desde la raíz del proyecto:

```powershell
python mt5_demo_executor_lab\mt5_demo_executor.py
```

## 4. Funcionamiento del Loop
El ejecutor realiza los siguientes pasos cada 60 segundos:
1. **Heartbeat:** Verifica conexión con MT5 y estado de la cuenta (aborto si es cuenta real).
2. **Timeout:** Cierra posiciones que lleven más de 4 horas abiertas.
3. **Scan:** Obtiene velas H1 y calcula niveles (Asia, London, PDH/PDL).
4. **Sweep:** Detecta si la vela H1 anterior barrió un nivel.
5. **Confirm:** Si hay sweep, busca reclaim en la vela M5 actual.
6. **Risk/News:** Valida que no haya noticias en ±30m y que el riesgo sea de 0.10%.
7. **Order:** Envía orden de compra/venta a la cuenta demo con SL y TP calculados.

## 5. Monitoreo
- **Logs:** Revisa `outputs/mt5_demo_log.csv` para eventos generales.
- **Trades:** Revisa `outputs/mt5_demo_telemetry.csv` para el registro de órdenes.
- **Status:** Revisa `outputs/mt5_demo_status.json` para ver el estado actual del loop.

## 6. Advertencia Institucional
Este laboratorio es **únicamente para pruebas técnicas en DEMO**. Cualquier intento de usar este código en cuentas reales sin la aprobación del tribunal y el cumplimiento de los gates shadow es una violación grave de los protocolos del proyecto.
