@echo off
REM SAP2 local batch GUI (VDA-style folder scan + engine per shot)
setlocal
cd /d "%~dp0"

if not defined SAP2_PYTHON (
  if exist "..\..\VideoDepthAnything_Portable\embedded_python\python.exe" (
    set SAP2_PYTHON=..\..\VideoDepthAnything_Portable\embedded_python\python.exe
  ) else (
    set SAP2_PYTHON=python
  )
)

echo ========================================
echo SAP2 Batch Processor
echo ========================================
"%SAP2_PYTHON%" sap2_batch_app.py
if errorlevel 1 pause
endlocal
