import pandas as pd
from pathlib import Path

def consolidate_rankings(root_dir):
    root = Path(root_dir)
    results = []
    
    # Buscar carpetas de ventanas
    windows = ["window_16_00", "window_16_30", "window_17_00"]
    
    for win in windows:
        win_path = root / win
        if not win_path.exists():
            continue
            
        # Buscar la subcarpeta de resultados más reciente (timestamped)
        subdirs = sorted([d for d in win_path.iterdir() if d.is_dir()], reverse=True)
        if not subdirs:
            continue
            
        latest_run = subdirs[0]
        ranking_file = latest_run / "strategy_ranking.csv"
        
        if ranking_file.exists():
            df = pd.read_csv(ranking_file)
            df["window"] = win
            results.append(df)
            
    if not results:
        print("No se encontraron rankings para consolidar.")
        return
        
    master_df = pd.concat(results, ignore_index=True)
    
    # Pivotar para comparar ventanas por estrategia
    comparison = master_df.pivot(index="strategy_name", columns="window", values="selected_score")
    
    output_path = root / "MASTER_CONSOLIDATED_RANKING.csv"
    master_df.to_csv(output_path, index=False)
    
    comparison_path = root / "WINDOW_COMPARISON_SCORE.csv"
    comparison.to_csv(comparison_path)
    
    print(f"Consolidación completada:")
    print(f"- Ranking maestro: {output_path}")
    print(f"- Comparativa de ventanas: {comparison_path}")

if __name__ == "__main__":
    target = "results/session_window_study_W1W2_CONSOLIDATED"
    consolidate_rankings(target)
