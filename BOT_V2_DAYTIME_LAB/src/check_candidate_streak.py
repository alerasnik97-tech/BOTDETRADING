
import pandas as pd

def check_streak(name):
    trades = pd.read_csv(rf"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase8_improvement_lab\final_combinations\{name}_trades.csv")
    results = trades['r_value'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)).tolist()
    max_loss = 0
    curr_loss = 0
    for r in results:
        if r < 0:
            curr_loss += 1
            max_loss = max(max_loss, curr_loss)
        else:
            curr_loss = 0
    print(f"Max Loss Streak for {name}: {max_loss}")

if __name__ == "__main__":
    check_streak("Candidate_B_F_Body60")


