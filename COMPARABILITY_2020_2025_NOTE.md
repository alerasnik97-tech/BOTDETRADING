# NOTA TÉCNICA OFICIAL: COMPARABILIDAD DE DATOS (2020-2021 vs 2022-2025)

## 1. Veredicto Oficial
**Estado**: `comparable_with_caveats` (Comparables bajo reservas).

## 2. Orígenes y Datasets Involucrados
* `data_free_2020`: Dataset inicial, abarca desde Enero 2020 hasta Diciembre 2021. Configurado con descargas amplias, no restringe la corrección de errores milimétricos (`strict_data_quality=False`).
* `data_candidates_2022_2025`: Dataset hiper-curado, abarca desde Enero 2022 hasta Diciembre 2025. Incluye `strict_data_quality=True`, protección contra shocks integrados (`shock_guard`), y una tolerancia inferior a spreads volátiles.

## 3. Comportamiento en Memoria (El Engine)
Ambos conjuntos convergen exitosamente dentro del `data_loader.py` bajo el método `load_prepared_ohlcv()`. A nivel bytes, concatenan de manera matemática permitiendo ejecutar Walk-Forward Analysis (WFA) ininterrumpido sobre la ventana bruta de 5 años.

## 4. Cuadro de Comparabilidad
**¿QUÉ SÍ DEBE COMPARARSE?:**
- Rendimiento base a nivel H1 y dirección macro de la estrategia. 
- Desgaste general de Equity Curves por periodos de regímenes laterales.
- Coeficiente Profit Factor macroscópico.

**¿QUÉ DEBE LEERSE CON EXTREMA CAUTELA? (El Caveat):**
- **Win Rates precisos en M5 (Microestructura):** Las ineficiencias de Spread del 2020 y 2021 podrían gatillar fills fantasmas (`optimistic fills`) debido a la falta de curaduría de la data pre-2022. 
- **Volatilidad post-Notice:** Los shocks noticiosos de 2022-2025 (Pospandemia, Guerra Ucrania, Inflación FED) no tienen análogo topológico exacto en 2020. 
- Una estrategia que sobrevive *exclusivamente* en 2020-2021 y se desploma en 2022-2025 **probablemente sea basura que dependa de ineficiencias de datos obsoletas o de un régimen de volatilidad Covid.**
- Una estrategia que fracasa *ligeramente* en 2020-2021 pero arrasa en 2022-2025 **debe ser promovida**, pues la validación más moderna y dura (curación explícita) ha triunfado.

**NUNCA SE DEBE:** Optimizar una estrategia para salvar su rendimiento pre-2022. Todo peso optimizador debe sesgarse al marco temporal validado y curado a posteriori.
