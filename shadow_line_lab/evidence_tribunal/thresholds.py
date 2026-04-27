# Thresholds Institucionales por Muestra (N)

GATES = {
    "N_0": { # Incubación inicial
        "min_trades": 0,
        "max_dd": 5.0,
        "min_pf": 0.0,
        "verdict": "SHADOW_INCUBATING"
    },
    "N_5": { # Alerta temprana
        "min_trades": 5,
        "max_dd": 8.0,
        "min_pf": 1.2,
        "min_expectancy": 0.1,
        "verdict": "SHADOW_HEALTHY_EARLY"
    },
    "N_10": { # Consistencia media
        "min_trades": 10,
        "max_dd": 10.0,
        "min_pf": 1.5,
        "min_expectancy": 0.2,
        "verdict": "SHADOW_HEALTHY_EARLY"
    },
    "N_20": { # Candidato a revisión de escalado
        "min_trades": 20,
        "max_dd": 15.0,
        "min_pf": 2.0,
        "min_expectancy": 0.3,
        "verdict": "SHADOW_ESCALATION_CANDIDATE"
    }
}

# Alertas de Seguridad
ALERT_CONFIG = {
    "max_consecutive_losses": 3,
    "max_timeout_rate": 0.40,
    "max_news_block_rate": 0.50,
    "max_no_execution_streak": 5
}
