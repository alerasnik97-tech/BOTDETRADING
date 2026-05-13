# RUNBOOK OPERATIVO

## Correr tests
```
cd BOT_V2_DAYTIME_LAB
.\venv_v37\Scripts\pytest.exe -o pythonpath=. src\v7_engine\tests\ -v
```

## Empaquetar ZIP liviano
```
python reports\v37_manipulante2\platinum_engine_certification\package_platinum.py
```

## Auditar ZIP
```python
import zipfile, hashlib
zf = zipfile.ZipFile("000_PARA_CHATGPT.zip")
print(zf.testzip(), len(zf.namelist()))
```

## Reglas inquebrantables
- NO tocar datos fuente (BOT_MARKET_DATA, data*, ticks)
- NO crear ZIP extra (solo 000_PARA_CHATGPT.zip)
- NO abrir Explorer automáticamente
- NO hacer git push sin autorización
- NO correr laboratorio sin autorización
- Si hay error, pausar y reportar
