# SOLICITUD DE CAMBIO AL MOTOR CENTRAL (ENGINE CORE CHANGE REQUEST)

*Instrucciones: Rellenar la totalidad de los campos descritos a continuación y someter a aprobación explícita del usuario/Dirección Quant antes de renombrar este archivo como `APPROVED_ENGINE_CORE_CHANGE_REQUEST.md` para desbloquear el pre-commit hook.*

## 1. Metadatos de la Solicitud
- **ID de Solicitud**: `ECC-YYYYMMDD-XXX`
- **Autor/Agente Proponente**: 
- **Fecha de Sometimiento**: 
- **Estado**: PENDING / APPROVED / REJECTED

## 2. Justificación Técnica
### Motivo Algorítmico o Arquitectónico
*[Explicar detalladamente por qué el cambio es absolutamente imprescindible para el proyecto institucional]*

### Alternativa Fuera del Core Considerada
*[Demostrar físicamente por qué no fue posible resolver esta necesidad aplicando el patrón Adapter/Wrapper en el espacio de nombres de la estrategia]*

## 3. Impacto en Código (Code Delta)
### Archivos Afectados
- `src/vX_.../archivo.py`

### Diff Esperado
```diff
- código original
+ código modificado
```

## 4. Análisis de Riesgos y Rollback
### Riesgos de Causalidad y Contabilidad
*[Describir posibles impactos sobre el cálculo de slippage, comisiones FTMO o filtración de información OOS]*

### Plan de Marcha Atrás (Rollback Plan)
*[Comandos exactos de Git para revertir al hash canónico anterior en caso de falla post-despliegue]*

## 5. Auditoría Criptográfica
- **Hash SHA256 Antes del Cambio**: 
- **Hash SHA256 Después del Cambio**: 

## 6. Plan de Certificación y Pruebas Obligatorias
- [ ] **Tests Targeted de la Capa Afectada**: `pytest src/vX/... -v`
- [ ] **Full Suite Institucional**: `pytest src/ -v` (100% Passed Requerido)
- [ ] **Smoke Test de Integridad**: Ejecución exitosa de `ENGINE_CORE_VERIFY.py` actualizando el manifiesto.

## 7. Firma de Aprobación
**Aprobación Explícita del Usuario/Director Quant**: 
- *Firma*: 
- *Timestamp*: 
