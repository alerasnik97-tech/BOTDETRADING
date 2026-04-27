@echo off
cd /d "C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\monitoring"
python "operational_monitor.py" >> "operational_logs\monitor_daily.log" 2>&1
exit /b %errorlevel%
