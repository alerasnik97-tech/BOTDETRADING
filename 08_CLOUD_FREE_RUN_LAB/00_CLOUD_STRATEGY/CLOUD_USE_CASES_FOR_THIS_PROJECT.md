# CLOUD_USE_CASES_FOR_THIS_PROJECT

Usar nube gratis para:
- micro-probes: validación de una sola configuración en un período corto.
- corridas nocturnas acotadas: backtests que duran 4-8 horas.
- validaciones por bloques: correr el mes X en una instancia y el mes Y en otra.
- tests livianos: integración de nuevos componentes.
- análisis de CSV: procesamiento de grandes outputs sin bloquear la CPU local.
- smoke tests: verificación rápida de que el runner no rompe en Linux.

No usar nube gratis para:
- datos privados pesados: información que no debe salir del entorno local.
- producción: no es un entorno confiable ni profesional para operar real.
- broker: riesgo de baneo o robo de credenciales.
- órdenes reales: falta de latencia controlada y estabilidad.
- secretos: nunca subir API keys o contraseñas.
- backtests sin checkpoint: pérdida de tiempo si la sesión se corta.
- sweeps sin límite: saturación de CPU y posible baneo del proveedor.
- TEST selection: elegir parámetros basados solo en resultados cloud sin validación local.
