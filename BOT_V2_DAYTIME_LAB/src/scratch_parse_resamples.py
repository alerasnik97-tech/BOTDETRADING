import os
from pathlib import Path
import re

src_dir = Path(r"c:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\src")

report = []
for root, _, files in os.walk(src_dir):
    for f in files:
        if not f.endswith(".py"): continue
        path = Path(root) / f
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()
            lines = content.split('\n')
            
            for i, line in enumerate(lines):
                if ".resample(" in line:
                    report.append({
                        "file": str(path.relative_to(src_dir)),
                        "line_num": i + 1,
                        "line_content": line.strip()
                    })

for r in report:
    print(f"FILE: {r['file']} | LINE: {r['line_num']} | CONTENT: {r['line_content']}")
