# Creates a LocalVoiceTranslator shortcut on the desktop.
$ErrorActionPreference = "Stop"

$appDir  = $PSScriptRoot
$pythonw = Join-Path $appDir "venv\Scripts\pythonw.exe"
$script  = Join-Path $appDir "app.py"
$desktop = [Environment]::GetFolderPath("Desktop")
$lnkPath = Join-Path $desktop "LocalVoiceTranslator.lnk"

if (-not (Test-Path $pythonw)) {
    Write-Host "pythonw.exe not found: $pythonw" -ForegroundColor Red
    Write-Host "Run setup.bat first, and run this script from the LocalVoiceTranslator folder." -ForegroundColor Yellow
    pause; exit 1
}
if (-not (Test-Path $script)) {
    Write-Host "app.py not found: $script" -ForegroundColor Red
    pause; exit 1
}

$sh  = New-Object -ComObject WScript.Shell
$lnk = $sh.CreateShortcut($lnkPath)
$lnk.TargetPath       = $pythonw
$lnk.Arguments        = "`"$script`""
$lnk.WorkingDirectory = $appDir
$lnk.WindowStyle      = 1
$lnk.Description      = "LocalVoiceTranslator - transcribe and translate speech locally"
$lnk.IconLocation     = "$pythonw,0"
$lnk.Save()

Write-Host ""
Write-Host "Shortcut created on your desktop:" -ForegroundColor Green
Write-Host "  $lnkPath"
Write-Host ""
Write-Host "Double-click 'LocalVoiceTranslator' on your desktop to start the app." -ForegroundColor Cyan
pause
