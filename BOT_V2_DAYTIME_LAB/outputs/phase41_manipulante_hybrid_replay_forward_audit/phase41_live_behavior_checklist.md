# PHASE41 LIVE BEHAVIOR CHECKLIST

| Regla | Validado (Replay) | Observacion |
| :--- | :--- | :--- |
| **Max 1 Trade/dia** | SÍ | El generador de señales filtra por `head(1)` diario. |
| **Ventana 07:00-16:30 NY** | SÍ | Respetada por el motor de señales. |
| **TP 1.4R** | SÍ | Aplicado en la simulacion de outcome. |
| **BE 0.4R** | SÍ | Aplicado en la simulacion de outcome. |
| **BF 70%** | SÍ | El motor de señales usa `body_pct >= 0.7`. |
| **Daily Close 19:45 NY** | SÍ | Implementado como overlay de seguridad. |
| **Friday Close 16:55 NY** | SÍ | Implementado como overlay de seguridad. |
| **News Gate** | LIMITADO | El motor offline marca limitacion por falta de cache historico. |
| **Data Quality Gate** | SÍ | Usa datos certificados con mascara de calidad. |
| **No duplica trades** | SÍ | El replay no genera trades encabalgados. |

## Conclusión
El comportamiento operativo del bot actual es **ALTAMENTE FIEL** a la estrategia auditada, con el agregado de capas de seguridad (cierres forzados) que mejoran la robustez para FTMO.
