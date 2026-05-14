# REPARACIÓN DE CRITERIOS DE SELECCIÓN DE CANDIDATOS (V49.5)
**Estrategia:** R1 (Mean Reversion / NY Open Absorption)  
**Estatus Post-Poda:** NO_VALID_CANDIDATES_AFTER_REPAIR  

---

## 1. Justificación de la Enmienda Metodológica
La auditoría independiente externa demostró que el pliego de selección de la versión V49 incurrió en un sesgo agudo al admitir configuraciones finalistas basándose de forma miope y exclusiva en el Profit Factor de la muestra de validación ($PF_{val}$), soslayando rentabilidades netas severamente negativas en la muestra de entrenamiento ($PF_{train} \in [0.51, 0.80]$). Para suprimir de raíz esta vulnerabilidad, se formaliza la adopción institucional del siguiente pliego restrictivo de maduración.

## 2. Nuevo Criterio Mínimo Contractual (Filtro de Supervivencia)
A partir de la presente fecha, una configuración queda **AUTOMÁTICAMENTE DESCALIFICADA** como candidata a finalista o pase a pre-producción si incurre en cualquiera de las siguientes 9 violaciones:

1. **Rendimiento In-Sample Deficitario:** $PF_{train\_net\_0.2} < 1.00$.
2. **Rendimiento de Validación Insuficiente:** $PF_{val\_net\_0.2} < 1.15$.
3. **Densidad Muestral TRAIN Pobre:** $N_{train} < 30$ transacciones.
4. **Densidad Muestral VAL Pobre:** $N_{val} < 20$ transacciones.
5. **Divergencia de Régimen Anómala:** Ratio de asimetría $PF_{val} / PF_{train} > 2.0$ carente de justificación microestructural fuerte.
6. **Parasitismo Transaccional:** El Top 5 de operaciones individuales explica más del **50%** del $PnL$ total en Validación.
7. **Concentración Mensual Extrema:** Un solo mes calendario concentra más del **60%** del $PnL$ total en Validación.
8. **Ausencia de Verificación de Degradación:** Carece de un barrido de estrés de slippage asimétrico de **0.3 pips** ejecutado sobre su propia grilla.
9. **Colisión de Firma Hash:** Es un duplicado exacto de otra configuración pre-existente evaluando su `trade_set_hash` determinista.

## 3. Aplicación de la Poda sobre el Universo V49
Al someter la nómina de finalistas y el Top 20 original a este nuevo embudo de cribado institucional, se constata la **DEPURACIÓN COMPLETA Y ABSOLUTA** del espacio de candidatos:
- Las Top 5 violan simultáneamente las reglas 1 ($PF_{train} < 1.0$), 5 (ratios de asimetría $> 3.3$) y 8 (carencia de estrés individualizado).
- Adicionalmente, las configuraciones `051` y `172` violan la regla 9 (duplicación por colisión de grilla).

$$\text{ESTADO FINAL} = \mathbf{NO\_VALID\_CANDIDATES\_AFTER\_REPAIR}$$

**Conclusión Contractual:** El laboratorio se encuentra desprovisto de configuraciones viables para la estrategia R1. Se impone el rediseño del espacio de búsqueda y la ejecución de barridos ampliados (Batch 4) tras enmendar los fallos de inyección paramétrica del motor.
