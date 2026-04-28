
# PHASE 22 FORWARD DEMO PROTOCOL

## 1. OBJETIVO (PAPER/DEMO ONLY)
Establecer el marco operativo para la simulación en tiempo real (Paper Trading) de la estrategia Phase 22 Auditada. **EL TRADING REAL ESTÁ ESTRICTAMENTE PROHIBIDO.**

## 2. REGLAS DE SEGURIDAD INSTITUCIONAL
- **Entorno**: Cuenta Demo únicamente.
- **MT5 Real**: Bloqueado.
- **cTrader/VPS**: Bloqueados por ahora (ejecución local o manual controlada).
- **Riesgo**: Lotaje demo mínimo/simbólico.
- **Sincronización**: Los trades deben ejecutarse exactamente según la señal del BOT V2.

## 3. PROTOCOLO DE DATOS
- **News Fortress**: Obligatorio (Fail-Closed). Si no hay ALLOW explícito, no se opera.
- **Data Quality Mask**: Obligatoria (Fail-Closed).
- **Slippage**: Máximo permitido en demo para validez: 0.5 pips.
- **Evaluación**: Mínimo 30 trades (Ideal 50) antes de cualquier revisión de ascenso.

## 4. INTEGRIDAD
- **Config Hash**: El operador debe verificar que el hash del archivo de configuración coincida con `phase22_forward_demo_config_hash.txt` antes de iniciar la sesión.
- **Kill Switch**: Si ocurre un error técnico o desviación, se detiene la demo inmediatamente.
