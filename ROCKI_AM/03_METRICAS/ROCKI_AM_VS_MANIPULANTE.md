# COMPARATIVA: ROCKI AM vs MANIPULANTE

Este documento compara el bot oficial actual (**MANIPULANTE**) con el candidato futuro (**ROCKI AM**).

| Caracteristica | MANIPULANTE (Oficial) | ROCKI AM (Futuro VPS) |
| :--- | :--- | :--- |
| **Horario (NY)** | 07:00 - 16:30 (Diurno) | 00:00 - 04:00 (Overnight) |
| **Gráfico** | M3 | M5 |
| **Logica** | H1 Fractal + M3 CHOCH | Sweep SCBI M5 |
| **Profit Factor** | 2.24 (Neto FTMO) | 2.44 (Baseline) |
| **Win Rate** | ~32.5% | **~62.1%** |
| **Expectancy** | 0.23R | **0.43R** |
| **Sample** | 2,625 trades | 1,080 trades |
| **Ejecucion** | Manual / Auto-Trial | Archivada / Futuro VPS |
| **Complejidad** | Media (Filtros de noticias) | Media (Filtros de liquidez) |
| **Estado** | **OPERATIVO** | **CONGELADO** |

## Conclusiones
1. **ROCKI AM** tiene metricas superiores en WR y Expectancy, lo que la hace ideal para una operacion de alta frecuencia nocturna donde el operador no puede estar presente.
2. **MANIPULANTE** es mas apto para la fase actual por su horario diurno y por tener una muestra historica mas amplia (Phase 27).
3. Ambos bots pueden coexistir en el futuro en una cuenta de fondeo, operando en sesiones diferentes para diversificar el riesgo de la curva de equidad.

**REGLA DE ORO**: No intentar "unir" las estrategias. Deben correr en procesos separados.
