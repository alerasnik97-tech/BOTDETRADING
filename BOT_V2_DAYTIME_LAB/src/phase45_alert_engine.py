import os
import json
import time
from datetime import datetime

class AlertEngine:
    def __init__(self, root_path):
        self.root_path = root_path
        self.obs_path = os.path.join(root_path, "MANIPULANTE", "16_OBSERVABILITY")
        self.health_json = os.path.join(self.obs_path, "daily", "latest_health_snapshot.json")
        self.quick_status_path = os.path.join(root_path, "MANIPULANTE", "10_LOGS_PAPER", "ftmo_trial_bot", "quick_status.txt")
        self.dashboard_path = os.path.join(root_path, "MANIPULANTE", "15_FORWARD_DEMO_SCORECARD", "FORWARD_DEMO_DASHBOARD.md")
        
    def load_json(self, path):
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None

    def read_quick_status(self):
        if not os.path.exists(self.quick_status_path):
            return {}
        data = {}
        try:
            with open(self.quick_status_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if '=' in line:
                        k, v = line.strip().split('=', 1)
                        data[k] = v
        except Exception:
            pass
        return data

    def detect_alerts(self):
        health = self.load_json(self.health_json)
        quick = self.read_quick_status()
        
        alerts = []
        now = datetime.now()
        timestamp_arg = now.strftime("%Y-%m-%d %H:%M:%S")
        
        # --- CRITICAL EVENTS ---
        
        # 1. BOT_APAGADO_DURANTE_SESION (If during session and runner not active)
        # Assuming session is roughly 08:00 - 18:00
        is_session = 8 <= now.hour <= 18
        if is_session and quick.get("RUNNER") != "ACTIVO":
            alerts.append(self.create_alert("CRITICAL", "BOT_APAGADO_DURANTE_SESION", 
                "Bot fuera de linea durante sesion operativa", 
                "Verificar runner.py y MT5.", "quick_status"))

        # 2. REAL_OR_EXNESS_DETECTED
        account = quick.get("CUENTA", "").upper()
        if "REAL" in account or "EXNESS" in account:
            alerts.append(self.create_alert("CRITICAL", "REAL_OR_EXNESS_DETECTED", 
                f"Cuenta Real/Exness detectada: {account}", 
                "DETENER BOT INMEDIATAMENTE. Cambiar a Demo.", "quick_status"))

        # 11. MT5_DISCONNECTED
        if quick.get("TERMINAL_CONNECTED") == "NO":
            alerts.append(self.create_alert("CRITICAL", "MT5_DISCONNECTED", 
                "MT5 desconectado del servidor", 
                "Revisar conexion internet o credenciales MT5.", "quick_status"))

        # 12. HEARTBEAT_STALE
        # We can check timestamp in quick_status
        last_upd = quick.get("ULTIMA_ACTUALIZACION_ARG", "")
        if last_upd:
            try:
                # Format is HH:MM:SS
                upd_time = datetime.strptime(last_upd, "%H:%M:%S").replace(year=now.year, month=now.month, day=now.day)
                diff = (now - upd_time).total_seconds()
                if diff > 300: # 5 minutes
                    alerts.append(self.create_alert("CRITICAL", "HEARTBEAT_STALE", 
                        f"Status stale hace {int(diff/60)} minutos", 
                        "Verificar si el bot se congelo.", "quick_status"))
            except:
                pass

        # 10. ORDER_SEND_ERROR
        if quick.get("ORDER_SEND") == "ERROR":
             alerts.append(self.create_alert("CRITICAL", "ORDER_SEND_ERROR", 
                "Error al intentar enviar orden", 
                "Revisar logs de MT5.", "quick_status"))

        # --- INFORMATIVE EVENTS ---
        
        # 1. BOT_STARTED / STOPPED would need state comparison. 
        # For now let's focus on state-based ones.

        # 3. BLOQUEADO_NOTICIAS
        if quick.get("NEWS") == "NO_TRADE":
            alerts.append(self.create_alert("INFO", "BLOQUEADO_NOTICIAS", 
                "Operativa bloqueada por noticias de alto impacto", 
                "Ninguna. El bot reanudara solo.", "quick_status"))

        # 7. TRADE_TAKEN_DEMO (Detect if open_position changed to YES)
        if quick.get("OPERACION_ABIERTA") == "SI":
             alerts.append(self.create_alert("INFO", "TRADE_TAKEN_DEMO", 
                "Posicion abierta detectada (Demo)", 
                "Monitorear evolucion.", "quick_status"))

        # 9. SAFE_TO_TURN_OFF_PC_YES
        if quick.get("SEGURO_APAGAR_PC") == "SI":
            alerts.append(self.create_alert("INFO", "SAFE_TO_TURN_OFF_PC_YES", 
                "Seguro apagar PC: No hay ordenes ni sesiones pendientes", 
                "Puede cerrar todo.", "quick_status"))

        return alerts

    def create_alert(self, severity, event_type, title, message, source):
        return {
            "timestamp_arg": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "severity": severity,
            "event_type": event_type,
            "title": title,
            "message": message,
            "recommended_action": message, # simplified for now
            "source": source,
            "dedup_key": f"{event_type}_{severity}"
        }

if __name__ == "__main__":
    # Test
    engine = AlertEngine(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
    alerts = engine.detect_alerts()
    print(json.dumps(alerts, indent=2))
