# MAPA DE ESTRATEGIAS - Sistema Actual vs Tu Lógica

## 🎯 Tu Estrategia Manual (EUR/USD)

### Lógica Operativa
| Componente | Descripción |
|------------|-------------|
| **Instrumento** | EUR/USD |
| **Horario** | 8:00 - 10:50 AM Nueva York |
| **Regla de trades** | 1 trade por día, 2da bala solo si 1ro es BE |
| **Contexto 1H** | Barridos de liquidez sobre extremos |
| **Extremos relevantes** | Asia, Londres, Día anterior, Semana/Mes anterior |
| **Entrada LTF** | Post-barrido, bajar a 3M |
| **Setup entrada** | CHoCH + FVG |
| **Gestión** | BE 1:1.2, SL en mitad mecha, TP 1:2.1 |

### Variantes a Estudiar
- A. Base: 3M CHoCH + FVG
- B. Comparativa: 3M vs 5M CHoCH + FVG
- C. Filtros: 15M, horario estricto, post-noticia roja
- D. Gestión: Distintos BE, distintos TP

---

## 🔧 Estrategias del Sistema (17 disponibles)

### MÁS CERCANAS A TU LÓGICA

#### 1. `strategy_ls_sr` - Liquidity Sweep SR
```python
Foco: Barrido de máximos/mínimos AM (07:00-11:00)
Timeframe: M5
Gatillo: Sweep + Rejection + Intention
SL: Mecha + buffer configurable
TP: 50% del rango AM
Gestión: Break-even configurable, max_hold
```
**Similitud**: 70% (barrido de extremos, SL en mecha)  
**Gap**: No CHoCH+FVG, no Asia/Londres/día previo

#### 2. `ny_br_pure` - NY Breakout Pure
```python
Foco: Breakout sobre niveles de lookback
Timeframe: M5
Gatillo: Breakout + Retest
SL: ATR-based
TP: R:R configurable (default 1.5, 2.0)
Sesión: 11:00-18:00 NY (ajustable)
```
**Similitud**: 60% (breakout de niveles)  
**Gap**: No es barrido de liquidez, no CHoCH+FVG

#### 3. `ny_br_ema` - NY Breakout + EMA Filter
```python
Foco: Breakout + filtro tendencia H1
Timeframe: M5
Filtro: EMA 50/200 en H1
Gatillo: Mismo que ny_br_pure
```
**Similitud**: 50% (tiene filtro HTF)  
**Gap**: No CHoCH+FVG, filtro diferente al tuyo

---

### OTRAS ESTRATEGIAS RELEVANTES

#### 4. `strategy_src` - Sweep Rejection Continuation
```python
Foco: Barrido + Rechazo + Continuación
Timeframe: M5
Complejidad: ALTA
Incluye: Sweep detection, rejection validation, momentum check
```
**Similitud**: 55% (barrido + rechazo)  
**Gap**: No CHoCH+FVG explícito

#### 5. `strategy_smr` - Bollinger Mean Reversion
```python
Foco: Reverión a media Bollinger
Timeframe: M5
Setup: Toque banda + RSI extremo
TP: Vuelta a media
```
**Similitud**: 20% (mean reversion diferente a tu momentum)

---

## ⚠️ GAPS CRÍTICOS IDENTIFICADOS

### Gap #1: CHoCH (Change of Character)
**Estado**: ❌ NO IMPLEMENTADO
**Descripción**: No hay detector de ruptura de estructura de mercado
**Impacto**: ALTO - Es tu gatillo principal
**Solución**: Implementar detector CHoCH en M5

### Gap #2: FVG (Fair Value Gap)
**Estado**: ❌ NO IMPLEMENTADO  
**Descripción**: No hay detector de imbalance de precios
**Impacto**: ALTO - Es tu zona de entrada
**Solución**: Implementar detector FVG en M5

### Gap #3: Timeframe 3M
**Estado**: ❌ NO DISPONIBLE
**Descripción**: Solo hay M5, M15, H1
**Impacto**: MEDIO - M5 puede ser suficiente
**Solución**: Evaluar si M5 es aceptable o preparar datos 3M

### Gap #4: Rangos Multisesión
**Estado**: ⚠️ PARCIAL
**Descripción**: Solo rango AM (07:00-11:00), no Asia/Londres/día anterior
**Impacto**: MEDIO - Reduce oportunidades válidas
**Solución**: Extender cálculo de rangos a otras sesiones

### Gap #5: Filtro Noticias Rojas
**Estado**: ❌ OFF FORZADO
**Descripción**: Sistema tiene noticias deshabilitadas por defecto
**Impacto**: BAJO-MEDIO - Tu lógica de "post-noticia roja" no aplica
**Nota**: Esto es intencional por riesgo DST

---

## 📊 RANKING DE ESTRATEGIAS POR PROXIMIDAD A TU LÓGICA

| Ranking | Estrategia | Similitud | Gap Principal | Recomendación |
|---------|-----------|-----------|---------------|---------------|
| 🥇 1° | `strategy_ls_sr` | 70% | No CHoCH+FVG | **Base para adaptar** |
| 🥈 2° | `ny_br_pure` | 60% | No es barrido de liquidez | Alternativa |
| 🥉 3° | `strategy_src` | 55% | Más compleja | Opcional |
| 4° | `ny_br_ema` | 50% | Filtro diferente | Descartar |
| 5° | Resto | <40% | No aplican | No usar |

---

## 🛠️ PLAN DE IMPLEMENTACIÓN

### Fase 1: Adaptar `strategy_ls_sr` (4-6 horas)
1. Agregar detector CHoCH en M5
2. Agregar detector FVG en M5
3. Modificar gatillo: CHoCH + FVG (no solo sweep+rejection)
4. Probar variantes de gestión (BE 1.2, TP 2.1)

### Fase 2: Extender Rangos (2-3 horas)
1. Calcular rango Asia (00:00-07:00 NY)
2. Calcular rango Londres (03:00-11:00 NY)
3. Calcular rango día anterior
4. Permitir barridos sobre cualquiera de estos niveles

### Fase 3: Validar Timeframe (2-4 horas)
1. Evaluar si M5 es suficiente vs necesitar 3M
2. Si se necesita 3M: preparar datos (downsample de M1 o tick)

### Fase 4: Filtros (2-3 horas)
1. Evaluar filtro 15M (rechazo)
2. Evaluar filtro horario más estricto (08:00-10:50)
3. Noticias: esperar solución UTC del sistema

---

## 📋 COMANDOS PARA EMPEZAR

### Correr estrategia más cercana (modo normal)
```bash
python run_canonical.py strategy_ls_sr normal
```

### Correr con variantes
```bash
python run_canonical.py ny_br_pure normal
python run_canonical.py ny_br_ema normal
```

### Correr laboratorio completo (lento)
```bash
python -m research_lab.main run-all --pair EURUSD --execution-mode normal --disable-news --max-evals 8 --seed 42
```

---

**Documento generado**: 2026-04-15  
**Para**: Análisis de estrategias EUR/USD  
**Siguiente paso**: Implementar CHoCH+FVG o probar adaptaciones de `strategy_ls_sr`
