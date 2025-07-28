@echo off
REM QXDM Log Analysis - Windows Test Script
REM ========================================

echo Starting QXDM Log Analysis Test...
echo.

REM Set test environment
set QXDM_ROOT=C:\Users\EEzim\Desktop\3gpp_Complexity\test_logs
set BLOCK_SIZE=200000
set BATCH_SIZE=256

echo Configuration:
echo   QXDM_ROOT = %QXDM_ROOT%
echo   BLOCK_SIZE = %BLOCK_SIZE%
echo   BATCH_SIZE = %BATCH_SIZE%
echo.

REM Create output directory for test results
if not exist "test_results" mkdir test_results

REM Run the analysis
echo Running QXDM analysis pipeline...
C:\Users\EEzim\Desktop\3gpp_Complexity\myenv\Scripts\python.exe ^
    src\tspec_metrics_2.py ^
    --checkpoint-file test_results\test_checkpoint.pkl ^
    --embeds-file test_results\test_embeddings.npz

echo.
echo ===============================================
echo Test completed! Check the following files:
echo   - release_metrics.csv
echo   - delta_metrics.csv
echo   - test_results\ (checkpoints)
echo ===============================================
pause
