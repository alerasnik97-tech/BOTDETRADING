# MANIPULANTE - PROMOTION GATE (REGRESO A CUENTA PAGA)

## Objetivo
Definir las condiciones objetivas y matematicas necesarias para comprar una cuenta FTMO paga tras la fase de validacion Forward Demo.

## 1. Requisitos Minimos Operativos (Mandatorios)
No se puede considerar una cuenta paga sin cumplir el 100% de estos puntos:
- [ ] **20 Trades Demo Reales**: Ejecutados por el bot actual en modo Demo/Trial.
- [ ] **0 Trades Fuera de Regla**: Ni uno solo.
- [ ] **0 Duplicados de Orden**: El sistema de bloqueo de duplicados debe haber sido 100% efectivo.
- [ ] **0 Fallos de Cierre 19:45 NY**: El bot debe haber cerrado todas las posiciones a la hora pactada.
- [ ] **0 Detecciones de Real/Exness**: El `account_gate` debe haber bloqueado intentos de conexion erroneos.
- [ ] **AutoTrading Estable**: El runner debe haber operado sin crashes durante al menos 2 semanas consecutivas.

## 2. Requisitos de Rendimiento (Robustez)
- [ ] **Slippage y Comision medidos**: Se debe haber auditado que los costos reales no destruyen el edge (Phase 38B logic).
- [ ] **BE Netos Monitoreados**: Confirmar que el bot gestiona los BE adecuadamente ante la comision de FTMO.
- [ ] **Expectancy Positiva**: El resultado neto de los 20 trades demo debe ser coherente con el backtest (considerando la varianza).

## 3. Bloqueos Absolutos (Abortar si ocurre)
- [ ] Modificacion manual de parametros de MANIPULANTE.
- [ ] Intervencion manual en trades abiertos (cerrar antes de tiempo, mover SL).
- [ ] Perdida de confianza en el sistema de logs.
- [ ] Fallos criticos en el News Gate que permitan operar durante noticias de alto impacto.

## Decision Final
Cuando se marquen todos los puntos de la seccion 1 y 2, se podra proceder a la compra del nuevo challenge de FTMO.

---
*Preservacion de Capital Primero.*
