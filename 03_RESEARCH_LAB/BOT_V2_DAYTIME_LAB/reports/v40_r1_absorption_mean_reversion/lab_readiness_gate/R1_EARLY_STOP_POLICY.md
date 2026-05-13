# CRITERIOS DE PARADA TEMPRANA Y MUERTE RÁPIDA (EARLY STOP POLICY)

## 1. Disparadores de Cancelación Inmediata (STOP_EARLY_RED)
La ejecución completa de los 76 meses deberá ser abortada incondicionalmente y la estrategia sellada con estado de rechazo si se verifica la ocurrencia de cualquiera de las siguientes anomalías durante el transcurso del backtest:

1. **Inviabilidad en Entrenamiento (TRAIN Gate)**: Al finalizar la partición de entrenamiento (2020-2021), el *Profit Factor Neto* (incluyendo comisiones y slippage de 0.2 pips) es inferior a `0.90` en la totalidad de las 54 configuraciones que posean significancia estadística (N > 30 trades).
2. **Quiebra de Cuentas (FTMO Blown)**: Ocurrencia masiva de violaciones de pérdida diaria máxima o pérdida máxima global en el simulador contable V7.
3. **Violación de Frecuencia**: Registro de una sola fecha calendario con más de 3 operaciones ejecutadas para una misma configuración.
4. **Contaminación por Truncamiento**: Inclusión indebida de operaciones etiquetadas como cierres de simulación a fin de mes (`EOM`) en el cómputo de las curvas de capital.
5. **Divergencia de Motor o Runner**: Detección de *drift* de firmas en los archivos congelados mediante monitoreo concurrente.
6. **Corrupción de E/S**: Errores repetidos de lectura/escritura superiores a 5 intentos consecutivos sobre las unidades de almacenamiento.

## 2. Condiciones de Permanencia (CONTINUE)
Se autoriza la prosecución ininterrumpida del cómputo siempre que:
- No se dispare ninguna de las guardas enumeradas anteriormente.
- El rendimiento del sistema se mantenga dentro de los márgenes de hardware estipulados en el *Resource Budget*.
- Las métricas de la estrategia continúen arrojando evidencia preliminar de captación de *edge* de absorción.

*Regla de Blindaje OOS: La partición de prueba (TEST) no podrá ser consultada, evaluada ni empleada para fundamentar una detención temprana, preservándose en aislamiento absoluto hasta la emisión del reporte cuantitativo final.*
