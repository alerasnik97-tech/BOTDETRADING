import json
with open(r'c:\Users\alera\Desktop\BOT DE TRADING CURSOR\reports\official_anchors\pipeline_run_report.json', 'r') as f:
    data = json.load(f)

print('BUILD STATS:')
print(json.dumps(data['build_stats'], indent=2))
print()
print('CONNECTORS:')
for c in data['connectors']:
    print("  - {}: {} ({} events)".format(c['id'], c['status'], c['events_emitted']))
