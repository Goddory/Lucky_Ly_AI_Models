$ErrorActionPreference = 'Stop'

$inner = "pkill -f '[u]vicorn api.main:app --host 0.0.0.0 --port 8000' || true"
wsl.exe -d Ubuntu -- bash -lc $inner
Write-Output 'stopped_or_not_running'
