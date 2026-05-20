@echo off
REM Push SAP2 engine + Colab scripts to apanner/sap2 (run from sap2 repo root)
setlocal
cd /d "%~dp0"

echo Adding deploy/, scripts/, batch/, colab_templates/, google desk refs...
git add deploy/external_engine/sap2_engine.py
git add scripts/sap2_colab_run.py scripts/sap2_infer.py scripts/sap2_models.py
git add scripts/exr_io.py scripts/plate_cache.py scripts/colab_setup.py
git add scripts/download_checkpoints_colab.py
git add launch_sap2_engine.bat batch/ colab_templates/ batch/sample_sap2_job.json
git add COLAB.md SAP2_COLAB_PLAN.md push_to_github.bat
git status
echo.
echo Review above, then commit:
echo   git commit -m "SAP2 Colab + local engine: EXR cache, high-res infer, batch GUI"
echo   git push origin main
pause
endlocal
