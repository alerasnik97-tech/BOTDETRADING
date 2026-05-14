# R1 LESSONS LEARNED

1. **La Muestra Representativa es Clave**: No es necesario correr 60 meses para saber si una familia estǭ muerta. Un barrido representativo de 10 meses ahorra semanas de cómputo.
2. **Cuidado con los Defaults del Motor**: El bug de `test_start_year` record la importancia de configurar explcitamente los parǭmetros de seguridad.
3. **El Batching no es opcional**: Para backtests pesados, la gestin de memoria (GC + batching) es la diferencia entre el Ǹxito y el crash.
4. **La Validacin IS/OOS es el Gatekeeper Definitivo**: No importa cuǭn bueno sea un resultado en VAL, si el TRAIN es dǸbil, la estrategia carece de robustez estructural.
5. **Deduplicacin Preventiva**: Invertir tiempo en sanear el grid de parǭmetros evita procesar configs redundantes.
