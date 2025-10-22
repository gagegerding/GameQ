@echo off
REM GameQ helper: build content then run the CLI game.
REM Usage: double-click this file, or run from a Command Prompt.
REM Assumes this .bat is located in the GameQ repo root.

setlocal enabledelayedexpansion

REM Change to the directory of this script
cd /d %~dp0

if not exist ".venv\Scripts\python.exe" (
  echo [!] Python venv not found at .venv. Create one first:
  echo     python -m venv .venv
  echo     .venv\Scripts\pip install -r requirements.txt
  pause
  exit /b 1
)

REM Activate venv
call .venv\Scripts\activate.bat

REM Build content (generates questions.json and circuit SVGs)
if exist "compiler\build_content.py" (
  echo [*] Building content...
  python compiler\build_content.py
) else if exist "build_content.py" (
  echo [*] Building content...
  python build_content.py
) else (
  echo [!] build_content.py not found (expected in repo root or compiler\).
  pause
  exit /b 1
)

REM Run the game
echo [*] Launching GameQ CLI...
python -m src.game

echo [*] Done.
pause
