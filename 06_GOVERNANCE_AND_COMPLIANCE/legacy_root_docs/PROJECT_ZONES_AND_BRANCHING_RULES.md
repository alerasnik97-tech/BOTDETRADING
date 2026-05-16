# PROJECT ZONES AND BRANCHING RULES

## 1. ZONA ROJA - PRODUCCION DEMO PROTEGIDA

La zona roja contiene MANIPULANTE y toda la superficie que puede afectar su forward demo. Durante research no se toca.

Incluye:

- `MANIPULANTE/`
- `MANIPULANTE/START_MANIPULANTE.bat`
- `MANIPULANTE/STOP_MANIPULANTE.bat`
- `MANIPULANTE/STATUS_MANIPULANTE.bat`
- scripts live de Phase37 y soporte FTMO Trial
- observability live de Phase44
- Telegram Alerts de Phase45
- Phase46 CI Safety salvo extensiones pasivas explicitamente aprobadas
- News Fortress
- Data Quality Mask
- configs live, caches live, tokens, secretos y `.env`

Regla: MANIPULANTE queda congelado como produccion demo. No se cambian TP, BE, BF, riesgo, horarios, maximo de trades, news gate, data-quality gate ni archivos de ejecucion operativa mientras arroja data forward.

## 2. ZONA AMARILLA - LABORATORIO

La zona amarilla es `LAB_STRATEGIES/`.

Uso permitido:

- ideas nuevas de estrategias EURUSD
- backtests de research cuando una fase futura lo autorice
- configs independientes de laboratorio
- reportes exploratorios
- templates de investigacion
- analisis de correlacion no operativo

Regla: las estrategias nuevas viven aca. No pueden depender de archivos live de MANIPULANTE ni modificar su ruta operativa.

## 3. ZONA VERDE - PORTFOLIO / PROMOTION

La zona verde es conceptual hasta que exista una estrategia validada sola.

Solo puede recibir estrategias que:

- pasaron validacion individual
- tienen reporte auditable
- fueron comparadas contra MANIPULANTE
- muestran correlacion menor a 0.5 contra MANIPULANTE antes de pensar en portfolio
- pasaron gates de promocion definidos antes de mirar resultados

Regla: portfolio no significa ejecucion real automatica. Cualquier promocion futura requiere revision, PR, CI verde y aprobacion explicita.

## 4. FLUJO DE TRABAJO CON GIT

`main` representa la superficie protegida de MANIPULANTE.

Reglas:

- no desarrollar estrategias nuevas directamente en `main`
- cada estrategia nueva usa una rama separada `research/<nombre_estrategia>`
- trabajar solo dentro de `LAB_STRATEGIES/` salvo una aprobacion explicita
- no usar `git add .`
- no usar `git commit -a`
- no usar `git push --force`
- usar `git add` selectivo por archivo o carpeta de laboratorio
- abrir PR/revision antes de merge
- Phase46 CI debe estar verde antes de merge

Comandos recomendados para revisar estado:

```powershell
git checkout main
git pull origin main
git status --short
```

Crear rama de research cuando se autorice una estrategia concreta:

```powershell
git checkout -b research/eurusd-daytime-orb-v1
```

Trabajar solo en una ruta aislada:

```text
LAB_STRATEGIES/EURUSD_DAYTIME/strategies/orb_v1/
```

Agregar selectivamente:

```powershell
git add LAB_STRATEGIES/EURUSD_DAYTIME/strategies/orb_v1/
git add LAB_STRATEGIES/EURUSD_DAYTIME/reports/<reporte>
```

Nunca:

```powershell
git add .
git commit -a
git push --force
```

## 5. FLUJO RECOMENDADO

1. Verificar que `main` este limpio.
2. Crear rama `research/<nombre_estrategia>`.
3. Trabajar solo dentro de `LAB_STRATEGIES/`.
4. Generar reporte de investigacion.
5. Ejecutar guard de aislamiento:

```powershell
python BOT_V2_DAYTIME_LAB/src/phase47a_lab_isolation_guard.py
```

6. Ejecutar CI local:

```powershell
python BOT_V2_DAYTIME_LAB/src/phase46_ci_safety_check.py
```

7. Usar `git status --short`.
8. Agregar cambios selectivamente.
9. No mergear sin aprobacion.

## 6. CONTRATO PARA EURUSD DAYTIME LAB

La linea futura se llama `EURUSD Daytime Lab`.

Parametros iniciales de research:

- par: EURUSD
- horario tentativo: 07:00-20:00 NY
- maximo futuro a investigar: 2 o 3 trades por dia
- primero validar cada estrategia sola
- despues comparar contra MANIPULANTE
- despues estudiar correlacion y portfolio

Esta fase no crea estrategias, no busca edge, no backtestea y no activa bots.
