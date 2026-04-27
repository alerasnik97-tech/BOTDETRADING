# MT5 Activation Steps: Recomendación Profesional

Dada la conclusión de la auditoría (`ONLY_SAFE_AS_MANUAL_OR_SEMI_AUTOMATIC`), se bloquea el despliegue de un bot 100% automático. El camino profesional es la **Ejecución Manual Controlada** o **Semiautomática**.

## Alternativa Recomendada: Micro-Piloto Manual Ultra Chico

### 1. Preparación de MT5 (Entorno Visual)
1.  **Timeframe:** Abrir dos gráficos de EURUSD: uno en **H1** (para sweeps) y otro en **M5** (para confirmaciones).
2.  **Niveles:** Dibujar manualmente o mediante un script de ayuda los niveles:
    - PDH/PDL (Día anterior).
    - Asia H/L (Rango 18:00 - 02:00 NY).
    - London H/L (Rango 02:00 - 08:00 NY).
3.  **Indicador de Noticias:** Tener una fuente externa (ForexFactory, etc.) para monitorear el filtro de ±30m.

### 2. Horario Operativo Sugerido (Ventana de Vigilancia)
Para maximizar la captura de los mejores setups detectados en el laboratorio (London y NY Early):
- **Encendido (Vigilancia):** 03:00 NY (Londres abierto).
- **Apagado (Vigilancia):** 12:00 NY (Londres cerrado, NY avanzado).

### 3. Pasos de Ejecución (Manual)
1.  **Vigilar H1:** Esperar a que una vela H1 haga sweep (mecha fuera, cierre dentro) de un nivel institucional.
2.  **Bajar a M5:** Si el sweep ocurre, buscar el "reclaim" (cierre M5 del lado correcto del nivel) dentro de la primera hora (+1h).
3.  **Ejecutar:** Abrir posición a mercado con SL en el extremo del sweep ±1 pip y TP a 1.5R.
4.  **Gestionar:** Poner una alarma a las 4 horas para cierre manual si el TP/SL no se ha tocado.

### 4. Por qué no Automatizar Hoy
Automatizar en MT5 hoy implicaría confiar en que el código del EA interpreta los domingos, las noticias y el timeout exactamente igual que el laboratorio, sin haber pasado por la fase de validación de ejecución (`N >= 10`).
