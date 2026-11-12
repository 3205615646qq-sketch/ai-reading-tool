@echo off
rem Single image reading: drag a jpg onto this bat, or run it and paste a path.
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
  echo Drag an image file onto this .bat, or paste an image path below:
  set /p IMG=Image path:
) else (
  set "IMG=%~1"
)
if not defined IMG (
  echo No image given.
  pause
  exit /b 1
)

rem drop quotes typed by hand, then normalize to an absolute path
set "IMG=%IMG:"=%"
for %%A in ("%IMG%") do set "IMG=%%~fA"

if not exist "%IMG%" (
  echo File not found: %IMG%
  pause
  exit /b 1
)

"%PY%" read_one.py --image "%IMG%"
echo.
pause
