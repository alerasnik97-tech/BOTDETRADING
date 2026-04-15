# Flujo de Trabajo Cloud / Multi-Entorno

Este documento define el flujo operativo estándar para trabajar con **BOT DE TRADING** entre el entorno local, el repositorio Git y la máquina virtual / nube de ejecución.

## Esquema de Entornos

1. **Local (Fuente de Verdad)**: `C:\BOT DE TRADING`
   * Aquí se desarrolla, se hacen pruebas pequeñas y se sella la infraestructura.
   * Es la única fuente autorizada para modificar código core.

2. **Git Privado (Sincronización e Historial)**: `GitHub / Git Privado`
   * Repositorio central que conecta Local y Nube.
   * Contiene solo código, configuración y datasets mínimos versionables (ej. CSVs condensados en `data_free_2020/prepared/`).

3. **Nube (Ejecución y Cómputo Pesado)**: `Cloud VM / Servidor`
   * Aquí se clona el repositorio desde Git.
   * Se usa exclusivamente para descargar data pesada (ticks, M1) y correr backtests exhaustivos de múltiples parámetros u optimizaciones largas.
   * Todo output de la nube (reportes, matrices de resultados) no se commitea a Git; se baja por SFTP o descarga directa.

4. **Backup Seguro**: `D:\BACKUPS\BOT DE TRADING`
   * Solo lectura. Backup periódico frío del entorno local entero.

5. **Intercambio IA (Handoff)**: `handoff\000_PARA_CHATGPT.zip`
   * Snapshot puntual para dar contexto a un asistente LLM. No es el repositorio.

---

## Reglas Operativas Obligatorias

* **Regla 1: Nunca editar en paralelo.** No escribas código localmente si dejaste cambios sin commitear en la nube, y viceversa.
* **Regla 2: Git es el puente.** Antes de mover trabajo a la nube, debes hacer `git add .`, `git commit` y `git push` desde local.
* **Regla 3: Pull manual en la Nube.** Al conectarte a la máquina en la nube, lo primero a ejecutar siempre es `git pull`.
* **Regla 4: Push desde la nube de forma excepcional.** Si en la nube escribes un script útil de análisis, commitealo rápido e inmediatamente haz `git pull` en local. No desarrolles features en la nube.
* **Regla 5: Data pesada ignorada.** La nube bajará los GBs de Dukascopy y tick data de forma local a la misma máquina virtual. Esos datos ignorados por `.gitignore` nunca vuelven al repo Git.

## Secuencia Típica de Sprint

1. **Local:** Modificas una estrategia y corres un `test_rejection_harness.py` local rápido.
2. **Local:** Haces commit: `git commit -am "Sprint 3: Nueva familia prev_day" && git push`
3. **Nube:** Entras por SSH/RDP. Ejecutas `git pull`.
4. **Nube:** Lanzas la corrida masiva: `python -m research_lab.main run-all ...` y te vas a dormir.
5. **Nube:** Al terminar, descargas el .csv de ranking y los JSONs de metadata usando SFTP o bajando el ZIP de outputs que generes manualmente. *No hacer commit de los resultados masivos.*
6. **Local:** Revisas los resultados y repites el ciclo.
