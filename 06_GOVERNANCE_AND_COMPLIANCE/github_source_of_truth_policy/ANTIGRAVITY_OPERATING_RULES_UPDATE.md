# ANTIGRAVITY OPERATING RULES UPDATE — MAY 2026

## 1. REGLA MAESTRA: GITHUB SOURCE OF TRUTH
Antigravity ya no debe usar ZIP como mecanismo principal de entrega.

## 2. Flujo Operativo Obligatorio
1. **Fase de Ejecución**: Generar reportes/CSVs/logs dentro de carpetas institucionales.
2. **Fase de Validación**: Ejecutar tests, engine verify, rowcount y auditorías de integridad.
3. **Fase de Sello**:
   - `git add` selectivo de archivos validados.
   - `git commit -m "[vXX/phase] descriptive message"`.
   - `git push origin clean-sync-branch`.
4. **Fase de Reporte**: Informar a ChatGPT mediante el formato de Handoff Policy.

## 3. Prohibiciones Estrictas
- **NO** generar `000_PARA_CHATGPT.zip` salvo orden explícita.
- **NO** dejar ZIPs en la raíz del proyecto.
- **NO** subir archivos ZIP al repositorio de GitHub.
- **NO** referenciar ZIPs como fuente de verdad en las respuestas.
- **NO** modificar la rama `main` sin autorización.

## 4. Autorizaciones
- Se permite y fomenta el uso de GitHub como única fuente de verdad para la revisión por parte de modelos de IA.
