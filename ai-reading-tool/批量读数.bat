@echo off
rem Batch reading: drag a folder of images onto this bat.
rem Results go to result.csv and vis\ inside this folder.
setlocal
cd /d "%~dp0"

rem ---- find a Python that has ultralytics installed ----
set "PY="
python -c "import ultralytics" >nul 2>nul && set "PY=python"
if not defined PY (
  py -3 -c "import ultralytics" >nul 2>nul && set "PY=py -3"
)
if not defined PY if exist "%USERPROFILE%\miniconda3\envs\aii\python.exe" set "PY=%USERPROFILE%\miniconda3\envs\aii\python.exe"
if not defined PY if exist "%USERPROFILE%\anaconda3\envs\aii\python.exe" set "PY=%USERPROFILE%\anaconda3\envs\aii\python.exe"
if not defined PY (
  echo No Python with ultralytics found. Install Python 3.10+, then run:
  echo     pip install ultralytics
  pause
  exit /b 1
)

if "%~1"=="" (
  echo Drag an image folder onto this .bat, or paste a folder path below:
  set /p DIR=Folder path:
) else (
  set "DIR=%~1"
)

if not defined DIR (
  echo No folder given.
  pause
  exit /b 1
)

rem drop quotes typed by hand
set "DIR=%DIR:"=%"

rem remove trailing backslashes WITHOUT ever writing a quote next to a
rem backslash: cmd reads \" as an escaped quote and swallows the closing one
set "DIR=%DIR%#"
set "DIR=%DIR:\#=%"
set "DIR=%DIR:#=%"
set "DIR=%DIR%#"
set "DIR=%DIR:\#=%"
set "DIR=%DIR:#=%"

for %%A in ("%DIR%") do set "DIR=%%~fA"

if not exist "%DIR%" (
  echo Folder not found: %DIR%
  pause
  exit /b 1
)

"%PY%" read_one.py --image "%DIR%" --save-csv result.csv
echo.
pause
