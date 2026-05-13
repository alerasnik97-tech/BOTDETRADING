import zipfile
import os
from pathlib import Path

base_dir = Path(r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo')
zip_path = base_dir / '000_PARA_CHATGPT.zip'

files_to_include = [
    # M4 Results
    '03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/v39_manipulante4_sweep_quality/',
    '03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/run_manipulante4_micro_probe.py',
    '03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/src/v7_engine/tests/test_manipulante4_sweep_quality.py',
    '03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/src/v7_engine/tests/test_manipulante4_displacement_gate.py',
    '03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/src/v7_engine/tests/test_manipulante4_guards.py',
    # Historical
    '03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/v38_manipulante3_htf_ltf/',
    '06_GOVERNANCE_AND_COMPLIANCE/artifact_delivery/single_zip_delivery_lock/FINAL_SINGLE_ZIP_VERIFICATION.txt',
]

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for item in files_to_include:
        item_path = base_dir / item
        if item_path.is_dir():
            for root, dirs, files in os.walk(item_path):
                for file in files:
                    full_path = Path(root) / file
                    # Skip big tick files if any accidentally in reports
                    if file.endswith('.parquet') or file.endswith('.h5'): continue
                    rel_path = full_path.relative_to(base_dir)
                    zipf.write(full_path, rel_path)
        elif item_path.exists():
            zipf.write(item_path, item_path.relative_to(base_dir))

print(f"ZIP created: {zip_path}")
