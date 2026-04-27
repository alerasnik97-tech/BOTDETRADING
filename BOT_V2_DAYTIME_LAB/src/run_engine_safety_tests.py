
import unittest
import json
import os
from datetime import datetime
from pathlib import Path

def run_safety_suite():
    print("EJECUTANDO ENGINE SAFETY TEST SUITE...")
    
    loader = unittest.TestLoader()
    test_dir = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\tests\engine_safety"
    suite = loader.discover(test_dir)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Process results
    total = result.testsRun
    failed = len(result.failures)
    errors = len(result.errors)
    passed = total - failed - errors
    
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "errors": errors
        },
        "details": {
            "failures": [str(f[0]) for f in result.failures],
            "errors": [str(e[0]) for e in result.errors]
        },
        "verdict": "ENGINE_SAFETY_GATE_PASSED" if failed == 0 and errors == 0 else "ENGINE_SAFETY_GATE_FAILED_REPAIR_REQUIRED"
    }
    
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\engine_safety_tests")
    
    # Save JSON
    with open(out_dir / "engine_safety_test_results.json", 'w') as f:
        json.dump(report_data, f, indent=4)
        
    # Save MD
    md_content = f"""# Engine Safety Test Results
    
**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Veredicto:** {report_data['verdict']}

## Resumen
- **Total Tests:** {total}
- **Pasados:** {passed}
- **Fallados:** {failed}
- **Errores:** {errors}

## Detalle de Fallos
"""
    if failed > 0 or errors > 0:
        for f in result.failures: md_content += f"- FAIL: {f[0]}\n"
        for e in result.errors: md_content += f"- ERROR: {e[0]}\n"
    else:
        md_content += "- Todos los tests de seguridad han pasado correctamente.\n"
        
    with open(out_dir / "engine_safety_test_results.md", 'w') as f:
        f.write(md_content)
        
    print(f"Pruebas completadas. Veredicto: {report_data['verdict']}")

if __name__ == "__main__":
    run_safety_suite()
