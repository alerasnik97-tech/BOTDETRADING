# V50B FAMILY WINDOW POLICY

**Objetivo**: Establecer los lmites operativos infranqueables para la investigación V50B.

## Poltica General
1. **Ventana NY Standard**: El trading estǭ restringido a la ventana de liquidez principal de Nueva York: **07:00 ?" 17:00 NY**.
2. **Prohibicin de Asia/Londres Temprano**: No se autorizan aperturas a las 03:00 NY (Londres Open) debido a la menor supervisin y el riesgo de volatilidad sin liquidez NY.
3. **Mǭximo de Trades**: 3 trades por da por configuracin.
4. **Filtro Macro**: Todas las familias (especialmente F12) deben respetar un buffer de noticias de alto impacto (predeterminado 5 min post-noticia en v7).

## Aplicacin por Familia
- **F01 (London Continuation)**: **RECHAZADA** en su forma actual (03:15 NY). Debe reescribirse para buscar continuaciones de Londres una vez abierta la sesión de NY (07:00+).
- **F06 (Volatility Breakout)**: **APROBADA** para ventana 07:00 ?" 12:00 NY.
- **F08 (Session Overlap)**: **APROBADA** para ventana 08:00 ?" 11:00 NY.
- **F12 (Macro Safe)**: **APROBADA** para ventana 09:00 ?" 12:00 NY con filtro macro real.

**Veredicto**: Poltica activa. F01 queda bloqueada hasta reescritura.
