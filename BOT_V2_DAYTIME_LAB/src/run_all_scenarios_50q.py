import subprocess
import os

SCRIPTS_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\src"
ENGINE_SCRIPT = os.path.join(SCRIPTS_DIR, "phase50q_independent_replay_engine.py")

scenarios = [
    ["--latency-seconds", "0", "--cost-r", "0.0"],
    ["--latency-seconds", "1", "--cost-r", "0.0"],
    ["--latency-seconds", "1", "--cost-r", "0.2"],
    ["--latency-seconds", "5", "--cost-r", "0.0"],
    ["--latency-seconds", "5", "--cost-r", "0.2"],
    ["--latency-seconds", "30", "--cost-r", "0.0"],
    ["--latency-seconds", "30", "--cost-r", "0.2"],
    ["--scenario", "NEXT_M1_OPEN", "--cost-r", "0.2"],
    ["--scenario", "WORST_5S", "--cost-r", "0.2"]
]

for args in scenarios:
    cmd = ["python", ENGINE_SCRIPT, "--audit"] + args
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("ERROR:", result.stderr)
