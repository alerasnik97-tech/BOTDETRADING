import os

path = r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\EURUSD_FAST_SIGNAL_25_ANNOTATIONS_BY_CHATGPT.csv'

print("=" * 60)
print("AUDITORÍA DEL INPUT FILE - RUTA CANÓNICA")
print("=" * 60)
print(f"RUTA: {path}")
print(f"EXISTE: {os.path.exists(path)}")

if os.path.exists(path):
    print(f"TAMANO: {os.path.getsize(path)} bytes")
    print(f"TIMESTAMP: {os.path.getmtime(path)}")
else:
    print("ARCHIVO NO ENCONTRADO")
