# INCUBATION READINESS POLICY

## Propósito
Definir el marco operativo y de gobernanza para la transición de estrategias desde la fase de Research Lab hacia Incubation Staging (Paper Trading / Demo).

## Alcance
Este protocolo aplica a cualquier estrategia que haya superado las fases de Backtest Validation y Test dentro de `03_RESEARCH_LAB`.

## Principios Fundamentales
1. **Cero Riesgo de Capital Real**: La incubación se realiza exclusivamente en entornos de simulación o cuentas demo.
2. **Fidelidad de Ejecución**: El objetivo primario es validar que el edge teórico se mantenga frente a la fricción real del mercado (spread, slippage, latencia).
3. **Integridad de Datos**: Los resultados de incubación deben registrarse en el Shadow Ledger de forma inmutable.
4. **Independencia de Research**: Una vez en incubación, los parámetros de la estrategia están congelados. Cualquier cambio requiere reiniciar el periodo de incubación.

## Responsabilidades
- **Research Agent**: Entrega el paquete de readiness y parámetros finales.
- **Incubation/Parallel Agent**: Prepara el entorno, supervisa la ejecución y audita los logs.
- **User**: Aprueba formalmente la entrada y salida de cada fase.
