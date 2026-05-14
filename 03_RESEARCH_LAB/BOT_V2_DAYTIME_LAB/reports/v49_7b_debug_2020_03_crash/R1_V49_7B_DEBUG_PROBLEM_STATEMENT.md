# R1 V49.7B DEBUG ?" PROBLEM STATEMENT

**Sntoma**: El runner `v49_7b_full_scope_runner.py` se detiene silenciosamente tras imprimir `Processing 2020-03...`. No genera errores en el log ni produce el archivo `R1_V49_7B_TRADES.csv`.

**Hechos**:
1. V49.7B fall en su primer mes de ejecucin.
2. 0 trades generados.
3. No hay procesos de Python activos tras la interrupcin.
4. El mes 2020-03 es conocido por su alta densidad de datos y volatilidad.

**Objetivo de Debug**:
Identificar el paso exacto donde el proceso muere (Carga de datos, Deteccin de niveles, Seales, Motor o Escritura) y verificar si es un problema de memoria o una excepcin no capturada.
