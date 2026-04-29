# PROJECT RESTRUCTURE - MANIPULANTE FINAL REPORT

## 1. Objetivo
Dejar la estrategia principal actual claramente identificada bajo el nombre **MANIPULANTE**, aislando la estructura operativa en una carpeta segura y archivando todos los baselines, variantes *shadow* y fases previas en la carpeta **ESTRATEGIAS**.

## 2. Estructura Creada
- `MANIPULANTE/`: Contiene la operativa diaria, runbooks, reglas de fondeo, MT5 Launcher seguro y checklists (Autoridad actual).
- `ESTRATEGIAS/`: Contiene los baselines (Phase18, Phase24), candidatos shadow (BE0.5), variantes rechazadas y archivadas (Phase19).
- `BOT_V2_DAYTIME_LAB/`: Laboratorio técnico intacto.
- `000_PARA_CHATGPT.zip`: Archivo ZIP canónico único con el estado final (4.67MB, 929 entradas).

## 3. Definición de MANIPULANTE
- **Carpeta:** `MANIPULANTE/`
- **Fuente:** Phase25 Authority
- **TP:** 1.4R
- **BE:** 0.4R
- **BF:** 70%
- **Global Hard Close:** SÍ (Viernes 16:55 NY, universal).
- **Autoridad:** SÍ, es la única estrategia aprobada.

## 4. Definición de ESTRATEGIAS
- **Carpeta:** `ESTRATEGIAS/`
- **Categorías:** 01_BASELINES, 02_CANDIDATOS_SHADOW, 03_RECHAZADAS, 04_ARCHIVADAS, 05_EXPERIMENTOS, 06_REPORTES_HISTORICOS, 07_NO_USAR_EN_REAL.
- **Shadow:** TP1.4_BE0.5_BF70.
- **Archivadas:** Phase19 (Lookahead bug).

## 5. Archivos Creados
Se crearon docenas de archivos. Los más destacados en la raíz:
- `ABRIR_MANIPULANTE_AQUI.txt`
- `ESTRUCTURA_DEL_PROYECTO.md`
- Actualización total de documentos maestros (`00_READ_THIS_FIRST.md`, `01_CURRENT_PROJECT_STATUS.json`, etc).

## 6. Archivos Movidos
- Se movió/renombró `Manipulante/` a `MANIPULANTE/`.

## 7. Validación de Consistencia
- Todas las estructuras requeridas y los documentos maestros existen y son coherentes.
- No hay duplicidad de ZIPs vivos.

## 8. ZIP Canónico
- **Ruta:** `000_PARA_CHATGPT.zip`
- **Tamaño:** 4,678,749 bytes
- **Entradas:** 929
- **SHA256:** `15f7f997234652dc357a56d08c8d9bdcb44b424b01aca96881d6d796a23a42c6`
- **Testzip:** Ningún error.
- **ZIP único vivo:** SÍ.

## 9. GitHub Sync
- **Commit:** SÍ (hash `8dd2191`).
- **Push:** SÍ (a `origin main`).
- **Force Push:** NO.

## 10. Riesgos
- `MANIPULANTE` no debe operarse en vivo/real hasta que se certifique y se apruebe el manual checkout. Actualmente está en Paper/Demo only.

## 11. Siguiente Paso Único
- **Operar MANIPULANTE en paper/demo con la regla global de cierre viernes 16:55 NY activa. No comprar evaluación real hasta completar manual checkout review.**
