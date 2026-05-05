import zipfile
import re

ZIP_PATH = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\000_PARA_CHATGPT.zip"
BANNED_PATTERNS = [
    r"\.env",
    r"secret",
    r"token",
    r"credential",
    r"password",
    r"mt5_local_config\.json"
]

def audit_zip():
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
            test_result = zf.testzip()
            if test_result:
                print(f"ZIP_CORRUPT: {test_result}")
                return

            print(f"ZIP_NAME: {ZIP_PATH}")
            print(f"FILE_COUNT: {len(zf.namelist())}")
            
            findings = []
            for name in zf.namelist():
                for pattern in BANNED_PATTERNS:
                    if re.search(pattern, name, re.IGNORECASE):
                        findings.append(f"NAME_MATCH: {name} (Pattern: {pattern})")
            
            if findings:
                for f in findings:
                    print(f)
            else:
                print("ZIP_AUDIT_PASS: No banned filenames found in ZIP structure.")

    except Exception as e:
        print(f"ZIP_AUDIT_ERROR: {e}")

if __name__ == "__main__":
    audit_zip()
