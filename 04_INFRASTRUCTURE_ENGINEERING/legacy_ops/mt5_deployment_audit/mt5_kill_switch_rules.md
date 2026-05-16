# MT5 Kill Switch Rules (Manual Piloto)

Cualquiera de estos disparadores obliga a la **desactivación inmediata** del micro-piloto manual en MT5 y el retorno a fase `SHADOW_ONLY`:

## 1. Triggers Técnicos
- **Desviación de Niveles:** Si los niveles de MT5 no coinciden con los del laboratorio post-Sunday Fix.
- **Error de Lote:** Si por error se abre un trade con riesgo > 0.25%.
- **News Breach:** Si se toma un trade dentro de la zona prohibida de noticias (±30m).

## 2. Triggers de Rendimiento
- **Drawdown > 5%:** Pérdida acumulada del capital asignado al piloto.
- **3 SL Consecutivos:** Pausa obligatoria de 48h para revisión de fidelidad.
- **Slippage Masivo:** Si la diferencia entre el precio teórico y el real supera los 2 pips de forma constante.

## 3. Triggers Psicológicos/Operativos
- **Sobre-operativa:** Realizar más de 1 trade por día calendario.
- **Indisciplina:** Mover el SL o TP fuera de las reglas del laboratorio.
- **Improvisación:** Tomar trades en niveles no autorizados o fuera de la ventana de confirmación.

---
**EL KILL SWITCH ES LA ÚNICA GARANTÍA DE SUPERVIVENCIA.**
