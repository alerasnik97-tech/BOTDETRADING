# PROTOCOLO DE ESCALAMIENTO Y RECOMENDACIÓN DE PRÓXIMA ACCIÓN

**Nivel de Alerta:** **CRÍTICO (CRITICAL)**  
**Decisión Directriz:** **DETENER AGENTE AFECTADO (STOP REQUIRED)**  

## Diagnóstico de Intervención

- **¿Conviene dejar seguir a los agentes?** **NO.** La continuidad sin corrección agrava la dispersión documental y compromete el sello de inmutabilidad del cierre de Gate 6.
- **¿Hay que pausar alguno?** **SÍ.** Se debe pausar inmediatamente las operaciones de escritura del **Agente 1 (Research)**.
- **¿Hay que pedir reporte parcial?** SÍ. Solicitar un reporte parcial justificando la alteración en curso del archivo ZIP y la invasión de la ruta de gobernanza.
- **¿Hay que resolver conflicto Git?** SÍ. Antes de autorizar cualquier commit, se deben deshacer o documentar formalmente las alteraciones sobre `000_PARA_CHATGPT.zip` y retirar del índice los archivos mal ubicados en `architecture/`.
- **¿Hay que esperar finalización?** NO. La intervención debe ser preventiva y quirúrgica en este instante.
- **¿Hay que generar ZIP?** **NO.**
- **¿Hay que NO generar ZIP?** **CONFIRMADO. PROHIBIDO GENERAR ZIP.** La preservación del estado aséptico impide que este agente o los otros generen empaquetados secundarios.

## Plan de Acción Inmediato para el Usuario

1. **Emitir Orden de Reubicación al Agente 1:** Instruir al Agente 1 para que mueva la subcarpeta `06_GOVERNANCE_AND_COMPLIANCE/architecture/manipulante3_htf_ltf_research/` hacia su entorno nativo en `03_RESEARCH_LAB/`.
2. **Auditoría Forense del ZIP Oficial:** Inspeccionar mediante `git diff` o herramientas locales por qué se encuentra modificado `000_PARA_CHATGPT.zip`. Si la modificación es espuria, ejecutar un `git checkout` o `git restore` sobre el ZIP y su archivo sha256.
3. **Desbloqueo Condicional del Agente 2:** Una vez purgado el directorio de gobernanza, el Agente 2 podrá crear su directorio `data_quality_audits/` y emitir sus reportes pasivos.
