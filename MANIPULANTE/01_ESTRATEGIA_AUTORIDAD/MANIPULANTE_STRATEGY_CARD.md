# MANIPULANTE STRATEGY CARD

## Identidad
- **Nombre de la Estrategia**: MANIPULANTE
- **Origen de Autoridad**: MANIPULANTE (Origen: Phase25)
- **Símbolo**: EURUSD
- **Estado**: CURRENT_AUTHORITY (Solo Paper/Demo)

## Parámetros Core
- **Contexto Operativo**: H1 Fractal Sweep
- **Gatillo de Entrada**: First M3 CHOCH
- **Filtro de Calidad**: Body Filter ≥ 70% (BF 70)
- **Timeframe de Entrada**: M3
- **Take Profit (TP)**: 1.4R
- **Break Even (BE) Trigger**: 0.4R

## Gestión de Riesgo y Horarios
- **Ventana Operativa**: 07:00 – 11:30 NY (Hora Local Nueva York)
- **Frecuencia**: Máximo 1 trade por día
- **Riesgo Recomendado Base**: 0.50% por operación

## Filtros de Seguridad (Kill Switches)
- **News Fortress**: FAIL-CLOSED (No operar sin confirmación ALLOW).
- **Data Quality Mask**: FAIL-CLOSED (No operar sin confirmación ALLOW).
- **Global Weekend Hard Close**: VIERNES 16:55 NY. Cierre obligatorio de toda posición. PROHIBIDO mantener operaciones durante el fin de semana. No hay excepciones manuales.

## Notas Adicionales
- La versión de TP 1.4R con BE 0.5R es exclusivamente un *shadow comparator* y no debe operarse con dinero (ni real ni en evaluación) como autoridad.
- Las órdenes NO se automatizan.
