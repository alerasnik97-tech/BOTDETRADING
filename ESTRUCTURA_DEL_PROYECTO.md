# ESTRUCTURA DEL PROYECTO

El proyecto se divide en 3 bloques lógicos estrictos para evitar confusiones operativas.

## 1. MANIPULANTE (La Autoridad)
Esta es la carpeta operativa diaria. Contiene la ÚNICA estrategia aprobada y validada para operar (en modo demo/paper). 
- Incluye el Launcher, las checklists, los templates, y la confirmación de la política estricta de **cierre todos los viernes a las 16:55 NY**.

## 2. ESTRATEGIAS (El Archivo / Museo)
Esta carpeta contiene todas las estrategias, fases, candidatos *shadow* (como BE0.5), variantes descartadas, y los baselines pasados (como Phase18 y Phase24).
- **NO DEBE OPERAR NADA** que esté en esta carpeta. Sirve exclusivamente como evidencia histórica y de respaldo de las investigaciones.

## 3. BOT_V2_DAYTIME_LAB (El Laboratorio Técnico)
Contiene los scripts (`.py`), configuraciones brutas, logs de validación profunda, reportes matemáticos y la base de simulación (Data Engineering, Monte Carlo, etc).
- Si usted es el operador/trader del día a día, no necesita tocar esta carpeta.

## 4. 000_PARA_CHATGPT.zip (El ZIP Canónico)
Este es el único archivo comprimido válido del proyecto. Se utiliza para sincronizar todo el conocimiento validado, excluyendo datos pesados (raw ticks) y secretos, para mantener un registro "portátil" y seguro del proyecto.

## LO QUE NO DEBE TOCAR
- No toque `manipulante_config.json` intentando cambiar TP o BE.
- No borre la carpeta `ESTRATEGIAS`.
- No abra cuentas reales.
- No modifique scripts en `BOT_V2_DAYTIME_LAB/src` a menos que sea en una nueva iteración de desarrollo validada.
