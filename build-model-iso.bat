@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo Python 3 was not found. Run this repository in WSL or install Python 3.
  exit /b 1
)
py -3 tools\build_litemodel_iso.py %*
if errorlevel 1 (
  echo.
  echo Windows note: GRUB ISO creation requires WSL with gcc, make, grub-pc-bin and xorriso.
  exit /b 1
)
endlocal
