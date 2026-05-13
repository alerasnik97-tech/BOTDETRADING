# RECONCILIACIÓN DE ARCHIVOS SUELTOS EN EL DIRECTORIO RAÍZ

**Fecha de Evaluación:** 2026-05-13  
**Objetivo:** Purgar el directorio raíz para restaurar el estándar de máxima higiene institucional.

## Inventario y Estado de Archivos Sueltos

1. **`LEER_PRIMERO_SUBIR_A_CHATGPT.txt`**
   - **Rol:** Instructivo de carga externa.
   - **Diagnóstico:** Archivo sidecar de soporte.
2. **`VERIFICACION_ZIP_CHATGPT.txt`**
   - **Rol:** Resumen de metadatos de empaquetado.
   - **Diagnóstico:** Archivo sidecar de soporte.
3. **`000_PARA_CHATGPT.sha256.txt`**
   - **Rol:** Firma de integridad del contenedor principal.
   - **Diagnóstico:** Sidecar directo del ZIP. Presenta estado modificado para coincidir con el nuevo hash `a98c55a3...`.
4. **`.pytest_cache`**
   - **Rol:** Directorio de memoria temporal generado por la suite de pruebas automatizadas.
   - **Diagnóstico:** Artefacto efímero de ejecución.

## Recomendación Normativa Esperada

De acuerdo con el protocolo de remediación, se dictaminan las siguientes acciones correctivas:

- **Migración de Sidecars:** Los archivos sidecar del ZIP (`LEER_PRIMERO_SUBIR_A_CHATGPT.txt`, `VERIFICACION_ZIP_CHATGPT.txt` y opcionalmente el sha256 si se integra en manifiesto) deben moverse al directorio `06_GOVERNANCE_AND_COMPLIANCE\artifact_delivery\` si ya no se desea su exposición visual directa en la raíz.
- **Tratamiento de Caché:** La carpeta `.pytest_cache` debe mantenerse declarada dentro de `.gitignore` (confirmado que ya lo está) y **solo podrá ser eliminada físicamente si no existe ningún proceso o runner activo** ejecutando pruebas en segundo plano.
- **Topología de Raíz Ideal:** El árbol principal debe decantar de forma estricta en:
  `7 Carpetas Canónicas` + `000_PARA_CHATGPT.zip` + `.gitignore` + `.git`

## Cláusula de Salvaguarda
**No se ejecuta ningún movimiento ni borrado en este instante** para evitar desestabilizar posibles lecturas concurrentes de los agentes en paralelo.
