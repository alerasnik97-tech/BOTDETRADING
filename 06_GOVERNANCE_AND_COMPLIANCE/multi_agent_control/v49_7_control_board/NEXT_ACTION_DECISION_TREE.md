# NEXT ACTION DECISION TREE — V49.7

## Criterios de Decisión

### 1. Resultados de Research (A1)
- **SI PF > 1.5 en TRAIN/VAL:** Promocionar a Cloud Run (A4) en Kaggle.
- **SI PF < 1.0:** Abortar rama de investigación, reevaluar hipótesis.
- **SI Inconcluso:** Ejecutar expansión de parámetros (micro-probe).

### 2. Auditoría de Datos (A2)
- **SI Integridad 100%:** Continuar con backtest v49.7.
- **SI Gaps Detectados:** Bloquear promoción a TEST hasta que los datos sean rectificados o el periodo sea excluido.

### 3. Coordinación Cloud (A4)
- **SI Paquete Validado:** Iniciar ejecución masiva en Kaggle.
- **SI Error de Dependencias:** Reconstruir `requirements-vps-optional.txt`.

---

## Próxima Acción Única Recomendada
**Sincronización de Gobernanza y Research:**
Tras la creación del Control Board, el Research Agent (A1) debe realizar un push de sus reportes parciales de `v49.7b` para que el Governance Agent (A3) pueda validar la ausencia de leakage antes de autorizar el Cloud Run masivo.
