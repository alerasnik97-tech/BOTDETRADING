# R1 DO NOT REPEAT PATTERNS

**PROHIBIDO REPETIR ESTOS COMPORTAMIENTOS EN FUTURAS FAMILIAS:**

1. **Aceptar PF_val alto con PF_train < 1.0**: Es un indicador casi garantizado de sobreajuste o suerte temporal.
2. **Ignorar la N (Número de Trades)**: Una muestra menor a 30 trades por fase es ruido.
3. **Concentracin Mensual > 60%**: Si un slo mes genera la mayora de la ganancia, el edge no es estadstico.
4. **Relajar el Gate por Ansiedad**: Las puertas de validacin estǭn para ser respetadas, no para ser "ajustadas" hasta que pase un candidato perdedor.
5. **Correr el Set de Prueba (TEST) para "ver si funciona"**: El TEST solo se abre para candidatos finales congelados. Abrirlo antes es contaminar el laboratorio.
6. **Usar Placeholders**: Cada nmero reportado debe tener un CSV de trades que lo respalde.
