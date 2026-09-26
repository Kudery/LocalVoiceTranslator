@echo off
setlocal
cd /d "%~dp0"

echo === LocalVoiceTranslator setup ===

REM Find Python via the Windows launcher, otherwise via python
where py >nul 2>&1
if %errorlevel%==0 (
    set "PY=py"
) else (
    where python >nul 2>&1
    if %errorlevel%==0 (
        set "PY=python"
    ) else (
        echo.
        echo Python not found. Install Python from https://www.python.org/downloads/
        echo and tick "Add python.exe to PATH" during installation.
        pause
        exit /b 1
    )
)

echo Python found: %PY%
REM Never re-create an existing venv: running 'venv' over an environment built with a
REM different Python version leaves incompatible packages behind (e.g. numpy cp312 on 3.14).
REM Upgraded Python? Close the app, delete the venv folder, then run setup.bat again.
if exist "venv\Scripts\python.exe" (
    echo Reusing existing virtual environment 'venv'.
) else (
    echo Creating virtual environment...
    %PY% -m venv venv
    if errorlevel 1 ( echo Creating the venv failed & pause & exit /b 1 )
)

echo Installing packages (this can take a while, ~1.5 GB)...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 ( echo Installation failed & pause & exit /b 1 )

echo.
echo === Done. Start the app with start.bat ===
pause
