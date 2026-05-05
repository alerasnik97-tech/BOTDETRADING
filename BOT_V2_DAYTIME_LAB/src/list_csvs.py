import os

start_dir = 'reports/manipulante_tick_historical'
for root, dirs, files in os.walk(start_dir):
    for file in files:
        if file.endswith('TRADE_LEVEL.csv'):
            print(os.path.join(root, file))
