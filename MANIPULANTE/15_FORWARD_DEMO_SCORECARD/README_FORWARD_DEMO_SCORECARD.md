# MANIPULANTE - FORWARD DEMO SCORECARD

## ¿Que es esto?
Es el sistema oficial de monitoreo de la etapa **Forward Demo** de MANIPULANTE. Su objetivo es medir la robustez operativa del bot en tiempo real antes de arriesgar capital en una cuenta paga.

## ¿Que mide?
- **Fidelidad Operativa**: ¿El bot hace lo que dice la estrategia?
- **Estabilidad Tecnica**: ¿Hay errores de conexion, duplicados o fallos de gates?
- **Costos Reales**: ¿Cuanto pagamos de comision y spread en Demo?
- **Disciplina**: ¿Respetamos el horario y el PC Off sin intervencion manual?

## ¿Que NO mide?
- **Edge Futuro**: Los resultados de ayer no garantizan los de mañana.
- **Psicologia Real**: Operar demo no es igual a operar real (aunque el bot lo haga por ti).

## Uso Diario
Cada dia, despues del cierre operativo (20:00 NY), se debe ejecutar el generador de scorecard para revisar el desempeño del dia.

## Condiciones de Bloqueo (Promotion Gate)
No se pasara a cuenta paga si existe:
- Un solo trade fuera de regla.
- Un solo fallo de News Gate no resuelto.
- Intervencion manual no documentada.
- Posiciones abiertas detectadas al apagar el PC.

---
*MANIPULANTE: Control, Disciplina y Preservacion.*
