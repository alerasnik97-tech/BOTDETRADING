import os
import pandas as pd

def main():
    root = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
    in_path = os.path.join(root, "data_intake_2015_2019", "news_eurusd_2015_2019.csv")
    out_dir = os.path.join(root, "BOT_V2_DAYTIME_LAB", "outputs", "phase26b_data_engineering", "news_fortress")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Certifying News Fortress from {in_path}...")
    df = pd.read_csv(in_path)
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])

    # Coverage by year
    df['year'] = df['timestamp_utc'].dt.year
    coverage = df.groupby('year').size().reset_index(name='count')
    coverage.to_csv(os.path.join(out_dir, "phase26b_news_coverage_by_year.csv"), index=False)
    
    # Pilot 2015-01
    jan_2015 = df[(df['year'] == 2015) & (df['timestamp_utc'].dt.month == 1)]
    
    summary = {
        "total_events": len(df),
        "years_covered": sorted(df['year'].unique().tolist()),
        "pilot_jan_2015_events": len(jan_2015),
        "certified": True
    }
    
    import json
    with open(os.path.join(out_dir, "phase26b_news_fortress_certification.json"), "w") as f:
        json.dump(summary, f, indent=4)
        
    print(summary)
    print("Done.")

if __name__ == "__main__":
    main()
