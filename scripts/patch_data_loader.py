from pathlib import Path

loader_path = Path("research_lab/data_loader.py")

target = """    # Inyeccion de Londres (03:00 - 11:00 NY)
    frame = _fill_fixed_range_columns(frame, "03:00", "11:00")
    # Inyeccion de Midday Range (11:00 - 13:00)"""

replacement = """    # Inyeccion de Londres (03:00 - 11:00 NY)
    frame = _fill_fixed_range_columns(frame, "03:00", "11:00")
    # Inyeccion de SB Anchor (03:00 - 08:30 NY)
    frame = _fill_fixed_range_columns(frame, "03:00", "08:30")
    # Inyeccion de Midday Range (11:00 - 13:00)"""

def main():
    content = loader_path.read_text(encoding="utf-8")
    if 'frame = _fill_fixed_range_columns(frame, "03:00", "08:30")' not in content:
        if target in content:
            content = content.replace(target, replacement)
        else:
            # Try with just the line
            content = content.replace('frame = _fill_fixed_range_columns(frame, "03:00", "11:00")', 
                                      'frame = _fill_fixed_range_columns(frame, "03:00", "11:00")\n    frame = _fill_fixed_range_columns(frame, "03:00", "08:30")')
    
    loader_path.write_text(content, encoding="utf-8")
    print("Patch applied to data_loader.py")

if __name__ == "__main__":
    main()
