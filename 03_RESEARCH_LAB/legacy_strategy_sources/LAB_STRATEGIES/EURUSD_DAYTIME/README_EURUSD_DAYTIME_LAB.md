# EURUSD DAYTIME LAB

Laboratorio aislado para investigar estrategias diurnas EURUSD en fases futuras.

Esta carpeta no contiene estrategias reales en Phase47A. Solo contiene estructura, templates y reglas de separacion.

## Alcance futuro

- Par: EURUSD
- Horario tentativo: 07:00-20:00 NY
- Research cap: 2 o 3 trades por dia
- Validacion individual antes de portfolio
- Comparacion posterior contra MANIPULANTE
- Correlacion objetivo contra MANIPULANTE menor a 0.5 antes de pensar en portfolio

## Carpetas

- `strategies/`: una subcarpeta por estrategia futura.
- `shared/`: utilidades comunes de laboratorio, no live.
- `reports/`: reportes exploratorios y evidencia de research.
- `correlation/`: analisis de correlacion posterior a validacion individual.
- `_templates/`: formatos base para research y configs.

## Reglas

- No tocar `MANIPULANTE/`.
- No tocar Phase37/44/45/46 salvo aprobacion explicita y pasiva.
- No tocar News Fortress.
- No tocar Data Quality Mask.
- No crear bots ejecutables.
- No crear scripts live.
- No correr MT5.
- No abrir cuentas reales ni demo operativa desde research.
- No ejecutar ordenes.
- No optimizar parametros en Phase47A.

## Flujo minimo para una estrategia futura

1. Crear rama `research/<nombre>`.
2. Crear carpeta bajo `strategies/<nombre>/`.
3. Copiar templates.
4. Completar hipotesis y reglas objetivas antes del backtest.
5. Ejecutar solo pruebas autorizadas.
6. Guardar reporte en `reports/`.
7. Correr el guard Phase47A.
8. Agregar cambios selectivamente.
