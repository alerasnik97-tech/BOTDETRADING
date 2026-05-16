import json
d = json.load(open('scratch/real_htf_filter_ab_results.json'))
t = d['rama_a']['trades']
print(f"N={len(t)}")
print(f"First: {t[0]['entry_time'][:10]}")
print(f"Last: {t[-1]['entry_time'][:10]}")
print(f"Keys: {list(t[0].keys())}")
# Sample trade
print(f"Sample: {json.dumps(t[0], indent=2)}")
