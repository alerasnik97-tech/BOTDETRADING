# CANONICAL ZIP OPERATING STANDARD

## 1. Qué es `000_PARA_CHATGPT.zip`
Es el archivo comprimido oficial, central y único que representa el estado institucional vigente, código y configuración del repositorio local de BOT V2 DAYTIME LAB para ser consumido por ChatGPT.

## 2. Por qué es el único ZIP canónico
Para evitar confusiones de versionado, pérdida de contexto, "cacheo" erróneo en la subida, o bifurcaciones de autoridad. Si existe un solo ZIP, siempre subiremos la versión más actual.

## 3. Qué debe incluir
- Documentos maestros (README, CURRENT_STATUS, AUTHORITY_MAP, ZIP_MANIFEST).
- Configuraciones de la Estrategia Autoridad (actualmente Phase 25).
- Reportes vigentes de cierres de Phase.
- Requerimientos activos de nuevas Phases (ej: Phase 26B).
- Checklist, outputs livianos, scripts operativos en `src/`.

## 4. Qué debe excluir
- Secretos, credenciales, tokens (`.env`, `keys`, `mt5_local_config.json`).
- Entornos y Git (`.venv`, `.git`, `__pycache__`).
- Archivos pesados y datos crudos (M1, M3 completos, tick data, `.bi5`, `.parquet`).
- Archivos temporales o `.zipbak`.

## 5. Cómo se reconstruye
Mediante un script de Python que empaqueta en un archivo temporal `000_PARA_CHATGPT.building` con reglas de exclusión estrictas y, si el empaquetado es exitoso, reemplaza el `000_PARA_CHATGPT.zip`.

## 6. Cómo se valida
El script debe abrir el ZIP resultante y comprobar:
- Que no hay errores (`testzip() == None`).
- Que contiene los documentos maestros y reportes cruciales.
- Que no contiene las rutas y extensiones prohibidas.

## 7. Cómo se maneja caché de ChatGPT
Si la plataforma "cachea" la carga de archivos o envía una versión antigua, se recomienda generar una **copia externa** temporal para bypass. 

## 8. Cómo se maneja una copia temporal fuera del proyecto
Se copia el ZIP finalizado al Escritorio o cualquier otra ubicación fuera de la raíz (ej: `Desktop\SUBIR_A_CHATGPT_TEMPORAL.zip`) y se sube ese archivo. Luego de la carga, se borra. Dentro del proyecto, siempre queda un único ZIP vivo.

## 9. Cómo se eliminan/neutralizan ZIPs duplicados
Se busca recursivamente `*.zip` y todo lo que no sea el ZIP canónico se le cambia la extensión a `.zipbak` y se envía a cuarentena.

## 10. Cómo se calcula SHA256
Al final del empaquetado, se lee el archivo `rb` y se calcula mediante la librería `hashlib` nativa.

## 11. Cómo se actualizan manifests
Cada reconstrucción del ZIP debe reescribir `ZIP_CONTENTS_MANIFEST.md` con los nuevos datos (entry count, SHA256, estado).

## 12. Qué hacer si ChatGPT recibe un ZIP viejo
Verificar que dentro del proyecto no haya otro ZIP vivo. Generar copia temporal externa (Punto 8) y reintentar.

## 13. Qué hacer si aparecen múltiples ZIPs
Ejecutar política de cuarentena (Punto 9) inmediatamente antes de proceder con el empaquetado final.

## 14. Qué hacer si el ZIP contiene data pesada
Eliminarla del ZIP, agregar metadata liviana `.md` que documente su ruta y tamaño, y reempaquetar.

## 15. Qué hacer si falta un reporte importante
Generar el reporte, confirmar su existencia y reempaquetar el ZIP. Nunca cerrar la validación si falta la autoridad o los handoffs críticos.
