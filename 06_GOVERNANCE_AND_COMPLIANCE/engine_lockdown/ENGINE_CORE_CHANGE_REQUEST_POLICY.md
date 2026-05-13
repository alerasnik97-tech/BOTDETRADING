# POLÍTICA DE GESTIÓN DE CAMBIOS DEL MOTOR (ENGINE CORE CHANGE REQUEST POLICY)

## 1. Regla Canónica de Modificación
**"Sin Change Request formalmente aprobado, queda absolutamente prohibido modificar el código del core."**

## 2. Procedimiento Operativo Obligatorio
Cualquier desarrollador, agente de investigación o ingeniero de infraestructura que pretenda alterar, parchar o extender un archivo perteneciente a las rutas protegidas (`src/v7_engine/` o `src/v6_utils/`) está forzado a adherirse a la siguiente secuencia:
1. Purgar cualquier modificación en el working tree y asegurar el estado inmutable mediante `git restore`.
2. Crear una copia de la plantilla oficial de solicitud de cambio (`ENGINE_CORE_CHANGE_REQUEST_TEMPLATE.md`).
3. Documentar rigurosamente la justificación técnica, los hashes de los archivos y demostrar la inviabilidad de usar el patrón Adapter.
4. Someter la solicitud a la Dirección Quant / Usuario.
5. Únicamente tras la firma afirmativa de aprobación en el documento, se procederá a guardar el archivo como `APPROVED_ENGINE_CORE_CHANGE_REQUEST.md` en la subcarpeta `engine_lockdown/`.
6. Realizar la modificación en el código y ejecutar la totalidad de las suites de prueba de regresión.
7. Al confirmar los cambios en Git, el hook local validará la licencia de excepción y admitirá el commit.
8. Una vez integrado el cambio, la licencia de excepción debe archivarse o invalidarse para restablecer el bastión de bloqueo incondicional.
