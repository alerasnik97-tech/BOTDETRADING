# RUTAS INSTITUCIONALES PROTEGIDAS (ENGINE CORE PROTECTED PATHS)

## 1. Directorios Protegidos (STRICT PROTECTED CORE)
La modificación directa de cualquier archivo dentro de las siguientes rutas está **estrictamente prohibida** para los agentes de investigación y scripts de optimización:

- `03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB\src\v7_engine\`
- `03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB\src\v6_utils\`

*Nota: La protección se extiende de forma recursiva a todos sus submódulos, utilidades y suites de pruebas asociadas.*

## 2. Espacios de Trabajo Permitidos para Estrategias (ALLOWED STRATEGY NAMESPACES)
El desarrollo algorítmico, detectores de señales, extracción de niveles y scripts orquestadores deben residir exclusivamente en las siguientes ubicaciones permitidas:

- `03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB\src\R1\`
- `03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB\src\strategies\`
- `03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB\run_*_micro_probe.py`
- `03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB\reports\`

## 3. Arquitectura de Extensión (Adapter Pattern Rule)
Si el diseño de una hipótesis requiere especializar el comportamiento de un componente interno de V6 o V7, **queda vetada la alteración del código original**. En su lugar, el desarrollador deberá aplicar el patrón de diseño *Adapter* o *Decorator*:
1. Crear una clase derivada o envoltorio dentro de `src/R1/` o `src/strategies/`.
2. Consumir la instancia original inmutable del core por composición o herencia limpia.
3. Orquestar la inyección de dependencias en el script `run_*_micro_probe.py` respectivo sin invadir el core.
