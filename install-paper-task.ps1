# Регистрирует задачу в Планировщике Windows: шаг пейпер-трейдинга каждые N минут.
# Запуск:  powershell -ExecutionPolicy Bypass -File .\install-paper-task.ps1
# Удаление: schtasks /Delete /TN "CryptoLab-PaperStep" /F

param([int]$Minutes = 15)

$ErrorActionPreference = "Stop"
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$bat = Join-Path $dir "run-step.bat"

if (-not (Test-Path $bat)) {
    throw "Не найден run-step.bat рядом со скриптом"
}

# проверяем, что Python вообще видит pandas — иначе задача будет молча падать
$python = $env:PYTHON
if (-not $python) { $python = "python" }
try {
    & $python -c "import pandas, numpy" 2>$null
    if ($LASTEXITCODE -ne 0) { throw "pandas не найден" }
} catch {
    Write-Warning "Python с pandas не найден по пути '$python'. Задайте переменную PYTHON."
}

schtasks /Create /TN "CryptoLab-PaperStep" /TR "`"$bat`"" `
    /SC MINUTE /MO $Minutes /F | Out-Null

Write-Host "Задача CryptoLab-PaperStep создана: каждые $Minutes мин."
Write-Host "Проверить:  schtasks /Query /TN CryptoLab-PaperStep /V /FO LIST"
Write-Host "Лог:        $dir\paper\step.log"
Write-Host "Статус:     python -m lab.paper status --alerts"
