# ANÁLISIS DE CONCENTRACIÓN DE EDGE POR SESIÓN (SESSION EDGE CONCENTRATION REPORT)

## 1. Desglose de Rendimiento Intradía (Subventanas NY)
El análisis forense de la distribución horaria de las 238 operaciones netas de la configuración líder arroja la siguiente topografía de desempeño:

| Subventana NY | Conteo ($N$) | Profit Factor Neto | Expectativa ($R$) | Drawdown ($R$) | Caracterización del Tramo |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **07:00 - 08:00** | 22 | 1.02 | +0.02 R | 2.50 R | Pre-apertura. Volumen intermedio, bajo impacto. |
| **08:00 - 11:00** | **142** | **1.31** | **+0.26 R** | **1.80 R** | **Apertura NY (Núcleo del Edge)**. Máxima eficiencia. |
| **11:00 - 13:00** | 38 | 1.10 | +0.08 R | 2.20 R | Almuerzo NY. Disminución de momento direccional. |
| **13:00 - 16:00** | 28 | 0.94 | -0.05 R | 3.10 R | Tarde NY. Ruido transaccional, reversiones espurias. |
| **16:00 - 17:00** | 8 | 0.85 | -0.15 R | 1.50 R | Cierre NY. Spreads crecientes, degradación por comisiones. |

## 2. Respuestas a Interrogantes del Protocolo
- **Mejor Profit Factor**: Se concentra de forma aplastante en el bloque de **08:00 a 11:00 NY** ($PF = 1.31$).
- **Mejor Expectativa**: Ídem, alcanzando **+0.26 R** netas por operación en la apertura.
- **Menor Drawdown**: Observado en el mismo bloque óptimo ($DD = 1.80 R$).
- **Zonas de Mayor Ruido**: La sesión vespertina (**13:00 a 17:00 NY**) aporta un notorio arrastre negativo, caracterizándose por cazar paradas de pérdida ante la falta de continuidad institucional.

## 3. Recomendaciones Arquitectónicas Futuras
- **¿Conviene operar 07:00-17:00 completo?**: NO. Mantener la ventana abierta durante la tarde erosiona parte de las ganancias acumuladas en la mañana.
- **¿Conviene acotar a 08:00-11:00 NY?**: **SÍ, ABSOLUTAMENTE**. El verdadero *edge* de absorción reside de forma casi pura en las primeras 3 horas de la sesión americana.
- **Impacto de la sesión PM**: Se concluye que la operatoria posterior a las 13:00 NY actúa como un destructor parcial del *edge*.

*Directiva de Inmutabilidad: De acuerdo con las normas de auditoría, estos hallazgos tienen carácter puramente consultivo y de recomendación para la fase de expansión paramétrica, quedando estrictamente prohibido alterar o recortar a posteriori las métricas oficiales del backtest actual.*
