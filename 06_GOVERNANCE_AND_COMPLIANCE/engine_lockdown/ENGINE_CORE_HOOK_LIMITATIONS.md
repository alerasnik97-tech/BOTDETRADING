# LÍMITES ARQUITECTÓNICOS DE LAS BARRERAS DE REPOSITORIO (HOOK LIMITATIONS)

## 1. Alcance Local del Pre-Commit Hook
El bastión técnico establecido en el archivo `.git/hooks/pre-commit` es un mecanismo de intercepción de eventos perteneciente en exclusiva al entorno de control de versiones de la máquina local.
- **Sin Versionado Directo**: Por el diseño fundamental de seguridad del sistema Git, los ganchos (hooks) no forman parte del árbol de seguimiento y **no se propagan ni se sincronizan de forma automática** al ejecutar operaciones de envío (push) o clonado (clone) hacia repositorios remotos como GitHub.

## 2. Impacto sobre Entornos Deslocalizados (Kaggle / Oracle Cloud)
Al instanciar el proyecto en una plataforma de cómputo deslocalizada o efímera (ej. *Kaggle Notebooks* o *Cloud Lab*), el contenedor clonará el código de la estrategia y el motor, pero **carecerá nativamente de la protección del pre-commit hook**, a menos que se inyecte un script explícito que copie la estructura del gancho al inicializar el entorno.

## 3. El Bastión Portable Definitivo
Debido a la limitación descrita en la capa de Git, el único y verdadero mecanismo de control de inmutabilidad **portable, distribuido e incondicional** es la dupla conformada por:
`ENGINE_CORE_VERIFY.py` + `ENGINE_CORE_HASH_MANIFEST.json`

- **Regla Obligatoria para Empaquetados en la Nube**: Toda exportación, contenedor Docker o paquete zip destinado a correr simulaciones en la nube tiene la obligación institucional imperativa de incluir la carpeta `engine_lockdown/` en su estructura, e invocar el script de verificación como primera instrucción bloqueante antes de habilitar la carga de parquets.
