@echo off
REM SAP2 Engine — local batch (job JSON). Uses SAP2_PYTHON or system python.
setlocal
set SAP2_ROOT=%~dp0
set JOB_FILE=%~1

if "%JOB_FILE%"=="" (
  echo Usage: launch_sap2_engine.bat "path\to\job.json"
  exit /b 1
)
if not exist "%JOB_FILE%" (
  echo ERROR: Job file not found: %JOB_FILE%
  exit /b 1
)

if not defined SAP2_PYTHON (
  if exist "%SAP2_ROOT%..\VideoDepthAnything_Portable\embedded_python\python.exe" (
    set SAP2_PYTHON=%SAP2_ROOT%..\VideoDepthAnything_Portable\embedded_python\python.exe
  ) else (
    set SAP2_PYTHON=python
  )
)

echo ============================================================
echo SAP2 ENGINE
echo Job: %JOB_FILE%
echo Python: %SAP2_PYTHON%
echo ============================================================
"%SAP2_PYTHON%" "%SAP2_ROOT%deploy\external_engine\sap2_engine.py" "%JOB_FILE%"
exit /b %ERRORLEVEL%
