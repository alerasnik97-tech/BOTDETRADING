## Estado
CLOUD_FREE_RUN_LAB_READY

## Resumen
Se ha establecido la infraestructura documental y estratégica para la ejecución de backtests en entornos de nube gratuitos. El laboratorio está blindado contra el uso de secretos, costos accidentales y conexión a brokers, priorizando la integridad de los resultados y la seguridad de los datos.

## Carpeta creada
C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\08_CLOUD_FREE_RUN_LAB

## Opciones gratuitas
- **Oracle**: VPS persistente para corridas nocturnas (Ranking 1).
- **Kaggle**: Sesiones de hasta 12h-30h con alta RAM para sweeps (Ranking 2).
- **Colab**: Prototipado rápido y visualización (Ranking 3).
- **GitHub Actions**: Integridad, tests unitarios y validación de paquetes (Ranking 4).

## Recomendación
Usar **Oracle Cloud Always Free** como primera opción para ejecuciones persistentes (overnight) debido a su naturaleza de VPS completo que permite `tmux` y persistencia de disco básica. **Kaggle** es la mejor alternativa para sweeps intensivos de memoria que no superen las 12 horas.

## Seguridad
- **secrets**: Bloqueados por política estricta y checklist.
- **datos**: Limitados a particiones necesarias y uso de datasets privados.
- **broker**: Prohibición absoluta de conexión y envío de órdenes.
- **costos**: Control mediante Free Tier Cost Guard y Billing Checklist.
- **ZIPs**: No se crean ZIPs internos; empaquetado externo fuera del proyecto.

## Próximo paso
Cuando se decida realizar la primera corrida en la nube, se deberá construir el `CLOUD_PACKAGE` siguiendo la especificación en `05_CLOUD_PACKAGE_STAGING_NO_ZIP` y realizar un preflight local.

## Prohibiciones respetadas
Confirmar:
- no código tocado: OK
- no runner tocado: OK
- no tests tocados: OK
- no datos tocados: OK
- no ZIP tocado: OK
- no backtest: OK
- no sweep: OK
- no cloud real: OK (solo preparación)
- no push: OK
- no Explorer: OK
