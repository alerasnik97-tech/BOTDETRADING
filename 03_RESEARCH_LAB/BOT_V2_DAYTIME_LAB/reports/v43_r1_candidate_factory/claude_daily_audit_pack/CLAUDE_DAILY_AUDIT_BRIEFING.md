# CLAUDE 4.7 OPUS HIGH AUDIT BRIEFING — R1 CANDIDATE FACTORY

## 1. Misión del Auditor
Se requiere un escrutinio forense implacable sobre los resultados de la fábrica de candidatos V43. Tu objetivo es **romper la estrategia**. Busca cualquier indicio de autoengaño, optimización espuria o fragilidad estructural.

## 2. Puntos de Control Obligatorios
- **Data Mining**: ¿Los 1200 escaneos generaron un "edge" que es solo ruido estadístico?
- **Test Leakage**: Revisa si hubo algún indicio de que los parámetros se ajustaron tras mirar la partición 2025-2026.
- **Concentración**: Audita si el PnL vive de 1-3 trades de cisne negro.
- **Slippage**: Evalúa si la degradación de 0.2 a 0.5 pips es demasiado agresiva.
- **Causalidad**: ¿La lógica de absorción matutina tiene sentido físico o es solo correlación accidental?

## 3. Veredicto Requerido
No uses lenguaje diplomático. Si encuentras un fallo crítico, reporta `R1_FACTORY_RED` y bloquea la promoción a la siguiente fase.
