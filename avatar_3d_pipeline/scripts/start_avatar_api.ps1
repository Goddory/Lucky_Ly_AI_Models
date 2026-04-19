param(
    [switch]$Strict,
    [switch]$PrintCommand,
    [int]$TextureSize = 256
)

$ErrorActionPreference = 'Stop'

function Convert-ToWslPath {
    param([Parameter(Mandatory = $true)][string]$WindowsPath)

    $full = [System.IO.Path]::GetFullPath($WindowsPath)
    $drive = $full.Substring(0, 1).ToLowerInvariant()
    $rest = $full.Substring(2).Replace('\', '/')
    return "/mnt/$drive$rest"
}

$projectWindowsPath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$projectWslPath = Convert-ToWslPath -WindowsPath $projectWindowsPath
$pythonExec = '/opt/miniconda/envs/avatar_env/bin/python'
$allowMissingPrefix = ''
$texturePrefix = "AVATAR_TEXTURE_SIZE=$TextureSize "

if (-not $Strict) {
    # Test-mode startup allows API boot when some model checkpoints are not present yet.
    $allowMissingPrefix = 'AVATAR_ALLOW_MISSING_WEIGHTS=1 '
}

$inner = "cd '$projectWslPath'; mkdir -p outputs/logs; if ss -ltn '( sport = :8000 )' | grep -q LISTEN; then echo already_running; else ${texturePrefix}${allowMissingPrefix}setsid -f $pythonExec -m uvicorn api.main:app --host 0.0.0.0 --port 8000 >> outputs/logs/server_autostart.log 2>> outputs/logs/server_autostart.log; echo started; fi"

if ($PrintCommand) {
        Write-Output $inner
}

wsl.exe -d Ubuntu -- bash -lc $inner
