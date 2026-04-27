import json
from pathlib import Path

manifest_path = Path('c:/Users/alera/Desktop/BOT DE TRADING CURSOR/data/official_anchors/manifests/user_curated_releases.json')
with open(manifest_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print('Current manifest structure (first 3 events):')
releases = data.get("releases", [])
for i, ev in enumerate(releases[:3]):
    print(f"Event {i}: {ev}")

print(f'\nTotal events: {len(releases)}')
print(f'Keys in manifest: {list(data.keys())}')
