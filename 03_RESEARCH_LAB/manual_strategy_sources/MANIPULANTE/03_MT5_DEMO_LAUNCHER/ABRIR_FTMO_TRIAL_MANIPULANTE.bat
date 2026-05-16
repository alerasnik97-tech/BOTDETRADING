@echo off
set ROOT=%~dp0..\..
cd /d "%ROOT%"
python BOT_V2_DAYTIME_LAB\src\phase37_ftmo_trial_bot_runner.py --ftmo-trial --dry-run --risk 0.005 --no-real
