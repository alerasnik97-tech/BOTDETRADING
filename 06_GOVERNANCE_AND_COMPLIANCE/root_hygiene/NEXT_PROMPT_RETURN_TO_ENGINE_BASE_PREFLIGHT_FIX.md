# RETORNO A ENGINE BASE PREFLIGHT FIX

Una vez confirmada la higiene estricta de la raíz (8 carpetas + .gitignore + .github), el siguiente paso es retomar la fase de **Engine Base & Preflight Fix**.

El objetivo es finalizar la remediación de los tests de preflight (leakage guards) y asegurar que el motor institucional esté listo para el gate final de apertura del laboratorio.

**Contexto a retomar:**
- `lab_preflight.py` requiere validación final contra datos reales.
- El motor tiene fijada la lógica de 1-bar offset y News Fortress.
- Los tests focalizados deben estar en verde antes de proceder.
