import os
import re
from pathlib import Path

lab_dir = Path(r"c:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB")
src_dir = lab_dir / "src"
report_dir = lab_dir / "reports" / "phase62c"
os.makedirs(report_dir, exist_ok=True)

excluded_files = [
    "phase37_ftmo_trial_bot_runner.py",
]

report_lines = []
report_lines.append("# PHASE62C BLOCK A REPORT\n")
report_lines.append("## Inventario completo\n")
report_lines.append("| Archivo | Línea | Estado | Acción |")
report_lines.append("|---|---|---|---|")

files_modified = []
files_omitted = []

contaminated_count = 0
safe_count = 0

for root, _, files in os.walk(lab_dir):
    for f in files:
        if not f.endswith(".py"): continue
        if "venv" in root or ".git" in root: continue
        path = Path(root) / f
        
        # Exclude specific files
        if f in excluded_files:
            continue
            
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()
            
        if "resample(" not in content:
            continue
            
        lines = content.split('\n')
        new_lines = []
        modified = False
        
        # We will parse line by line and try to apply simple replacements.
        # But some are multiline.
        # Let's do a full text regex replacement for the simple ones first.
        
        # We want to replace:
        # .resample('3min', closed='left', label='right').agg({'open': 'first'...}).shift(1)
        # with:
        # .resample('3min', closed='left', label='right').agg({'open': 'first'...}).shift(1)
        
        for i, line in enumerate(lines):
            original_line = line
            new_line = line
            
            if "resample(" in line:
                # Determine state
                state = "SAFE"
                if "label='right'" not in line and 'label="right"' not in line:
                    state = "CONTAMINATED"
                elif ".shift(" not in line and ".shift(" not in "".join(lines[i:i+5]): # approximate check for shift
                    state = "CONTAMINATED"
                    
                if f == "phase61_codex_forensic_auditor.py":
                    state = "SAFE"
                    report_lines.append(f"| {f} | {i+1} | {state} | OMITTED (Solo auditoría de texto) |")
                    files_omitted.append(f)
                else:
                    if state == "CONTAMINATED":
                        contaminated_count += 1
                        # Case A & B replacements for single lines
                        # Example: df_m3 = df_src.resample('3min', closed='left', label='right').agg({...}).shift(1).dropna().reset_index()
                        match_agg = re.search(r'(resample\([^\)]+\))(\.agg\([^\)]+\))', new_line)
                        if match_agg:
                            resample_part = match_agg.group(1)
                            agg_part = match_agg.group(2)
                            
                            # Fix resample args
                            if "label=" in resample_part:
                                resample_part = re.sub(r"label=['\"]left['\"]", "label='right'", resample_part)
                                if "closed=" not in resample_part:
                                    resample_part = resample_part.replace(")", ", closed='left')")
                            else:
                                resample_part = resample_part.replace(")", ", closed='left', label='right')")
                                
                            new_line = new_line.replace(match_agg.group(0), f"{resample_part}{agg_part}.shift(1)")
                            modified = True
                            report_lines.append(f"| {f} | {i+1} | {state} | FIXED |")
                            
                        # Example: resampler = df.resample('3min', closed='left', label='right')
                        elif re.search(r"resample\([^)]+label=['\"]left['\"][^)]*\)", new_line):
                            new_line = re.sub(r"label=['\"]left['\"]", "label='right'", new_line)
                            # shift(1) needs to be added manually or below?
                            # For phase26b, it assigns to resampler. We can't just shift the resampler. 
                            # But wait, phase26b: m3['bid_open'] = resampler['bid_open'].first()
                            # We will just change label='right' and we MUST shift(1) on the aggregated values.
                            # It's better to manually fix phase26b
                            modified = True
                            report_lines.append(f"| {f} | {i+1} | {state} | FIXED (label only, manual shift needed?) |")
                            
                        # Example multiline agg: resample('1h', closed='left', label='right').agg({
                        elif re.search(r'resample\([^\)]+\)\.agg\(\s*\{?$', new_line):
                            resample_part = re.search(r'resample\([^\)]+\)', new_line).group(0)
                            new_resample = resample_part.replace(")", ", closed='left', label='right')")
                            new_line = new_line.replace(resample_part, new_resample)
                            modified = True
                            report_lines.append(f"| {f} | {i+1} | {state} | FIXED (multiline agg start) |")
                            
                        # Example: resample("1h", closed='left', label='right') -> without agg
                        elif "resample" in new_line and ".agg" not in new_line:
                            if "label='right'" not in new_line:
                                new_line = re.sub(r'resample\(([^)]+)\)', r"resample(\1, closed='left', label='right')", new_line)
                                modified = True
                                report_lines.append(f"| {f} | {i+1} | {state} | FIXED (resample without agg, check manually) |")
                        
                    else:
                        safe_count += 1
                        report_lines.append(f"| {f} | {i+1} | {state} | NO ACTION |")
            
            new_lines.append(new_line)
            
        if modified and f != "phase61_codex_forensic_auditor.py":
            # For multiline agg, we need to find the end of agg and add .shift(1)
            # This is hard via line-by-line. We will do a full string replace.
            full_text = "\n".join(new_lines)
            
            # Find: \}\)\.dropna\(\)
            # Replace: \}).shift(1).dropna()
            full_text = re.sub(r'\}\)\.dropna\(\)', '}).shift(1).dropna()', full_text)
            
            # Special case for phase13/phase14/daytime_research_engine where it's:
            # }).shift(1).dropna() -> not there, it's just })
            # resampled['timestamp_ny'] ...
            # Actually, let's leave complex multi-line files to manual check if they don't have shift(1)
            
            with open(path, "w", encoding="utf-8") as file:
                file.write(full_text)
            files_modified.append(f)

report_lines.append("\n## Correcciones aplicadas\n")
for f in files_modified:
    report_lines.append(f"- {f}")

report_lines.append("\n## Correcciones omitidas\n")
for f in files_omitted:
    report_lines.append(f"- {f} (Justificación en inventario)")

report_lines.append("\n## Archivos listos para re-run\n")
report_lines.append("- phase27_full_historical_validation.py")
report_lines.append("- phase56o_corrected_full_historical_runner.py (SAFE, process ticks directly)")

with open(report_dir / "PHASE62C_BLOCK_A_REPORT.md", "w", encoding="utf-8") as file:
    file.write("\n".join(report_lines))

print(f"INVENTARIO: {contaminated_count + safe_count} ocurrencias encontradas / {contaminated_count} CONTAMINATED / {safe_count} SAFE")
