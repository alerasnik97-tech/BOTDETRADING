# ANTIGRAVITY OPERATING RULES UPDATE — MAY 2026

## 1. REGLA MAESTRA: GITHUB SOURCE OF TRUTH
Antigravity ya no debe usar ZIP como mecanismo principal de entrega.

## 2. Flujo Operativo Obligatorio
1. **Fase de Ejecución**: Generar reportes/CSVs/logs dentro de carpetas institucionales.
2. **Fase de Validación**: Ejecutar tests, engine verify, rowcount y auditorías de integridad.
3. **Fase de Sello**:
   - `git add` selectivo de archivos validados.
   - `git commit -m "[vXX/phase] descriptive message"`.
   - `git push origin <RAMA_CANONICA_VIGENTE>` (non-force; actualmente `governance/root-hygiene-20260516` o rama derivada curada). **NUNCA** `git push origin clean-sync-branch` como flujo operativo; **nunca** force push; **nunca** `main` sin autorización.
4. **Fase de Reporte**: Informar a ChatGPT mediante el formato de Handoff Policy.

## 3. Prohibiciones Estrictas
- **NO** generar `000_PARA_CHATGPT.zip` salvo orden explícita.
- **NO** dejar ZIPs en la raíz del proyecto.
- **NO** subir archivos ZIP al repositorio de GitHub.
- **NO** referenciar ZIPs como fuente de verdad en las respuestas.
- **NO** modificar la rama `main` sin autorización.
- **NO** tratar `clean-sync-branch` como rama operativa/source-of-truth: es una **historical donor branch / unrelated orphan history** sin ancestro común con la línea canónica. Integración solo curada (file-level port / cherry-pick bajo owner approval), nunca direct merge ni `--allow-unrelated-histories`.

## 4. Autorizaciones
- Se permite y fomenta el uso de GitHub como única fuente de verdad para la revisión por parte de modelos de IA.
