# DECISIÓN INSTITUCIONAL FINAL SOBRE LA ARQUITECTURA DE RAMAS (BRANCH FINAL DECISION)

## 1. Justificación del Uso de `clean-sync-branch`
La consolidación de los desarrollos recientes de inmutabilidad sobre la rama `clean-sync-branch` obedece a una necesidad irrenunciable de eficiencia en la sincronización. Dicha rama representa el nuevo inicio canónico (surgical clean start), desvinculado de objetos blob históricos pesados y resultados masivos de Manipulante 4, lo que la convierte en el vehículo óptimo para la distribución.

## 2. Estatus como Rama Oficial para Despliegues en la Nube
**`clean-sync-branch` queda designada oficialmente como la rama canónica temporal y primaria para clonación en Kaggle y Cloud Lab.**
- Al retener un tamaño mínimo y contener las guardas de bloqueo de core plenamente operativas, garantiza que los entornos efímeros arranquen rápidamente y sin arrastrar el lastre histórico del repositorio de investigación completo.

## 3. Disposición sobre la Rama Histórica
La rama pre-existente `agent/research-r1-absorption-mean-reversion` **no es destruida ni eliminada**, pero queda catalogada formalmente en estado de **Congelamiento y Desuso Activo (Deprecated Read-Only Vault)**. 
- Retiene valor puramente como registro de la evolución temprana de la estrategia R1 antes de la imposición del bloqueo de inmutabilidad del core.

## 4. Protocolo Anti-Confusión y Directiva de Clonado
Para prevenir ambigüedades entre los investigadores y agentes automatizados, se establece la siguiente pauta:
- **Desarrollo Local y Futuros Agentes**: Deben posicionarse exclusivamente en `clean-sync-branch` y construir sus commits incrementales sobre esta base.
- **Directiva de Clonado para Kaggle**: El archivo de configuración de inicialización en la nube o script bash debe apuntar incondicionalmente a clonar esta rama específica:
  ```bash
  git clone --branch clean-sync-branch --single-branch <URL_DEL_REPOSITORIO>
  ```
