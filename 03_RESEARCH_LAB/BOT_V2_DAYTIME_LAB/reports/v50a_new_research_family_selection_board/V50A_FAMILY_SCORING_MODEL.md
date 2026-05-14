# V50A FAMILY SCORING MODEL

El modelo de scoring asigna una puntuacin de 0 a 5 a cada familia basǭndose en criterios cualitativos y cuantitativos ponderados.

## Criterios y Ponderacin
| Criterio | Peso | Descripcin |
| :--- | :--- | :--- |
| **Causalidad Clara** | 2.0x | ŋHay una explicacin lógica y económica para el edge? |
| **Diferencia vs Manipulante** | 2.0x | ŋEs un concepto operativo distinto al actual? |
| **Bajo Riesgo Overfit** | 2.0x | ŋPocos parǭmetros y lógica robusta? |
| **Auditabilidad** | 1.5x | ŋEs fǭcil verificar los trades y la lógica? |
| **Frecuencia Esperada** | 1.0x | ŋGenera suficientes trades para ser estadsticamente vǭlida? |
| **Costo Computacional** | 1.0x | ŋVelocidad de procesamiento de los backtests? |
| **Compatibilidad EURUSD** | 1.0x | ŋSe adapta bien al par principal? |

## Frmula
`Score Final = sum(Puntuacin * Peso) / sum(Pesos)`

## Umbrales
- **Score > 4.0**: Prioridad Alta (Shortlist Inmediata).
- **Score 3.0 - 4.0**: Prioridad Media (Evaluacin Secundaria).
- **Score < 3.0**: Rechazada (Archivada).
