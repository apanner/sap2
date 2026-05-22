@echo off
REM Download SAP2 checkpoints to Google Drive VDA_models (for Colab + Desk)
setlocal
cd /d "%~dp0"

set ROOT=%~1
if "%ROOT%"=="" set ROOT=G:\My Drive\VDA_models\sapiens2_host
if not exist "%ROOT%" (
  echo Trying default user Drive path...
  set ROOT=%USERPROFILE%\Google Drive\VDA_models\sapiens2_host
)

echo ========================================
echo SAP2 model download
echo Target: %ROOT%
echo ========================================
echo.

set PYTHON=python
if exist "..\..\VideoDepthAnything_Portable\embedded_python\python.exe" (
  set PYTHON=..\..\VideoDepthAnything_Portable\embedded_python\python.exe
)
if exist "..\..\google_desk_app\python_embedded\python.exe" (
  set PYTHON=..\..\google_desk_app\python_embedded\python.exe
)

"%PYTHON%" -m pip install -q huggingface_hub
"%PYTHON%" download_checkpoints_colab.py --root "%ROOT%" --which matting_1b normal_1b

echo.
echo Done. Expected files:
echo   %ROOT%\matting\sapiens2_1b_matting.safetensors
echo   %ROOT%\normal\sapiens2_1b_normal.safetensors
pause
endlocal
