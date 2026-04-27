# POLÍTICA DE PROMOCIÓN DE ESTRATEGIAS (Strategy Promotion Policy)

Este documento define la taxonomía rigurosa y no negociable para clasificar la viabilidad de cualquier idea evaluada en este laboratorio. Todo sistema nace en `HARD_REJECT` hasta que demuestre lo contrario.

## 1. ESTADOS POSIBLES

### Nivel 0: `HARD_REJECT` (Rechazo Absoluto)
* **Significado:** Estrategia aniquilada matemáticamente. No presenta Edge (Ventaja) o sufre un colapso inaceptable (PF negativo, Drawdowns destructivos, Expectancy nula bajo spread).
* **Acción:** Morir. Fin del análisis. NO se optimiza para salvarla. 

### Nivel 1: `SOFT_REJECT` (Rechazo Flexible)
* **Significado:** La estrategia sobrevive con Expectativa positiva pero rompe algún umbral secundario (ej: Max Drawdown > 15%, o consistencia temporal irregular con meses perdidos acumulados).
* **Acción:** Archivo vivo condicional. Demuestra que "la idea tiene sentido", pero como producto aislado es demasiado inestable para dinero real. Merece *Discovery lateral* (ej: inyectarle un filtro HTF o usarla como filtro para otra).

### Nivel 2: `PASS_MINIMUM` (Aprobación Básica)
* **Significado:** Ha superado limpiamente la evaluación In-Sample (IS) y también resistió la prueba ciega Out-Of-Sample (OOS) mediante Walk-Forward Analysis, cumpliendo requisitos justos de supervivencia (Profit Factor > 1.0, DD < 15%).
* **Acción:** Sobreviviente. Se la enlista como **`research_candidate`**.
* **Advertencia:** **ESTO NO ES UN EDGE CONFIRMADO PARA PRODUCCIÓN.** Pasar la mínima con *Execution Normal* significa que el algoritmo no hace locuras matemáticas; pero su Edge podría diluirse bajo latencia. 

### Nivel 3: `STRONG_CANDIDATE` (Candidato Sólido)
* **Significado:** Superó la evaluación OOS y lo hizo con métricas holgadas: PF_OOS > 1.25, Constancia superior (0 o muy bajos años negativos), Expectancia amplia, y superviviendo bajo modo `stress` o `precision`. 
* **Acción:** Esta estrategia merece ser sometida a stress-tests exóticos (noticias críticas, Monte Carlo simulado mental, slippage por encima del histórico). 

### Nivel 4: `LIVE_CANDIDATE` (Candidato para Producción Incubada)
* **Significado:** Es un `STRONG_CANDIDATE` que fue auditado visualmente por el trader en la carpeta *PARA CHATGPT*, superó inspecciones de código, y generó consenso total de ejecución bajo Spread Real en cuentas micro.

## 2. REGLAS DE PROMOCIÓN (CRITERIOS INCUESTIONABLES)
1. **De `PASS_MINIMUM` a `STRONG_CANDIDATE`:** 
   No es un "score" mágico. Se asciende **solamente si se re-ejecuta la misma estrategia (vía `run_canonical.py precision`) y el Performance WFA retiene PF > 1.15 y no excede Drawdown del 12%.** Requiere una micro-fase secundaria que certifique el fill del spread intradiario.
2. **De `STRONG_CANDIDATE` a `LIVE_CANDIDATE`:**
   Requiere inyección del módulo estricto de Noticias (`DEFAULT_NEWS_V2_UTC_FILE`) activado, y una revisión discrecional del desarrollador asegurando que no compra absurdamente antes del FOMC o del NFP. 

## 3. ADVERTENCIAS METODOLÓGICAS (ANTI-DELIRIOS)
* Un *Pass Minimum* con Drawdown de 14.9% no es un "sistema increíble a punto de hacerte rico", es una bomba matemática pendiendo de un hilo temporal fino. Cautela ante la euforia.
* Optimizaciones consecutivas para rescatar una estrategia del Nivel 1 (`SOFT_REJECT`) al Nivel 2 incurren en pecado de *Overfitting*. Si tu familia madre falla reiteradamente, debes abandonar la hipótesis central y saltar a otro Sprint.
