$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
Set-Location $Root
python BOT_V2_DAYTIME_LAB\src\phase37_ftmo_trial_bot_runner.py --ftmo-trial --dry-run --risk 0.005 --no-real
