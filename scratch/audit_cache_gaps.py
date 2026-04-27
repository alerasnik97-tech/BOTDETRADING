import pandas as pd
import json
from pathlib import Path

def audit_cache():
    cache_path = Path("data/forex_factory_cache.csv")
    output_path = Path("data/official_anchors/manifests/curated_supplementary_us.json")
    
    if not cache_path.exists():
        print(f"Error: {cache_path} not found")
        return

    print(f"Loading {cache_path}...")
    df = pd.read_csv(cache_path)
    
    # Critical families
    TARGET_FAMILIES = ["Retail Sales m/m", "Unemployment Claims", "Core Retail Sales m/m"]
    
    # Filter for target families and USD currency
    mask = (df["Event"].isin(TARGET_FAMILIES)) & (df["Currency"] == "USD")
    subset = df[mask].copy()
    
    print(f"Found {len(subset)} raw events for target families.")
    
    events_list = []
    for _, row in subset.iterrows():
        # Promotion logic: Treat all as HIGH for safety, regardless of source label
        # The 'DateTime' in cache is often ambiguous (e.g. 00:00:00+03:30)
        # We will parse the DATE and assume 08:30 NY time if it matches the pattern
        
        try:
            raw_dt = pd.to_datetime(row["DateTime"])
            date_str = raw_dt.strftime("%Y-%m-%d")
            
            event = {
                "event_name": row["Event"].lower(),
                "currency": "USD",
                "release_date": date_str,
                "release_time_ny": "08:30",  # Standard for these families
                "impact_level": "HIGH",       # PROMOTION: Fortress safety first
                "source": "audited_local_cache",
                "verification_status": "promoted_from_legacy"
            }
            events_list.append(event)
        except Exception as e:
            print(f"Error parsing row: {e}")

    # Deduplicate by (event_name, date)
    unique_events = {}
    for ev in events_list:
        key = (ev["event_name"], ev["release_date"])
        unique_events[key] = ev
        
    final_events = list(unique_events.values())
    final_events.sort(key=lambda x: (x["release_date"], x["event_name"]))
    
    print(f"Generated {len(final_events)} unique, promoted events.")
    
    # Save to JSON
    output_data = {
        "manifest_name": "Curated Supplementary US Releases",
        "description": "Critical news families promoted from local cache for AM session safety.",
        "release_count": len(final_events),
        "events": final_events
    }
    
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Successfully saved to {output_path}")

if __name__ == "__main__":
    audit_cache()
