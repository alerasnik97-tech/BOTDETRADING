# MANIPULANTE - LECCIONES APRENDIDAS (KNOWLEDGE BASE)

Este documento destila el conocimiento adquirido durante el desarrollo, auditoria y despliegue del bot **MANIPULANTE**.

## 1. El Peligro del PF Bruto
Nunca se debe declarar una estrategia como "lista" basandose solo en el Profit Factor bruto de un backtest.
- **Leccion**: El edge debe ser auditado trade-por-trade contra comisiones, spreads variables y slippage real.

## 2. Auditoria Neta Obligatoria
- **Leccion**: Todo futuro bot debe pasar por una auditoria de costos (Phase 38B) antes de considerarse candidato para fondeo. No se puede pasar por alto el impacto del lotaje en las comisiones fijas.

## 3. El Impacto de los Break-Even (BE)
MANIPULANTE genera muchos BE. En el backtest bruto parecen "0R", pero en la realidad son pequenas perdidas por comision.
- **Leccion**: Se debe calcular el "BE neto" para entender si el sistema sigue siendo defendible bajo alta frecuencia de BE.

## 4. Seguridad Operativa Ante Todo
- **Leccion**: Impedir runners duplicados y tener un boton de STOP inteligente es mas importante que ganar un 1% extra. La preservacion del capital y de la cuenta de fondeo es la prioridad #1.

## 5. Separacion de Capas
- **Leccion**: Mantener separada la Logica de Trading de la Logica de Ejecucion (Runner) y del Overlay Operativo (Filtros de noticias/data). Esto permite auditar cada parte sin romper el sistema completo.

## 6. Disciplina de Parametros
- **Leccion**: Una vez que un bot es declarado oficial y "LOCKED", no se deben modificar parametros (TP/BE/BF) por ansiedad ante una racha negativa. Solo una auditoria forense completa puede justificar un cambio.

## 7. News Fortress es Vital
- **Leccion**: Operar noticias de alto impacto sin un gate automatico es un riesgo inaceptable para una Prop Firm. El modo FAIL-CLOSED es la configuracion estandar.

---
*"La excelencia no es un acto, sino un habito de auditoria constante."*
