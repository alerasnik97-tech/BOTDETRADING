# Thresholds de Checkpoints (Conservadores)

CHECKPOINT_RULES = {
    5: {
        "name": "Lectura Exploratoria",
        "min_trades": 5,
        "min_expectancy": 0.1,
        "max_dd": 8.0,
        "min_pf": 1.2
    },
    10: {
        "name": "Primera Señal de Consistencia",
        "min_trades": 10,
        "min_expectancy": 0.2,
        "max_dd": 10.0,
        "min_pf": 1.5
    },
    20: {
        "name": "Tribunal de Escalado",
        "min_trades": 20,
        "min_expectancy": 0.3,
        "max_dd": 15.0,
        "min_pf": 2.0
    }
}

# Perfil Esperado del Candidato (Baseline Research)
# Para comparación Observed vs Expected
CANDIDATE_BASELINE = {
    "pf": 2.4,
    "expectancy_r": 0.41,
    "max_dd": 15.0 # En % o R, según la escala usada
}
