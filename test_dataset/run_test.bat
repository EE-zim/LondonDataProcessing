@echo off
echo ====================================
echo 简化版QXDM测试数据集处理器
echo Simplified QXDM Test Dataset Processor  
echo ====================================

cd /d "%~dp0"

echo.
echo 当前目录: %CD%
echo.

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python环境
    echo 请确保Python已安装并添加到PATH
    pause
    exit /b 1
)

echo Python环境检查通过

REM 检查必要的目录
if not exist "QXDM_Logs_Sample" (
    echo 警告: 未找到QXDM_Logs_Sample目录
)

if not exist "test_logs_sample" (
    echo 警告: 未找到test_logs_sample目录  
)

echo.
echo 运行测试处理脚本...
echo.

python test_processor.py

if errorlevel 1 (
    echo.
    echo 处理过程中出现错误
    pause
    exit /b 1
)

echo.
echo ====================================
echo 处理完成！
echo.
echo 生成的文件:
if exist "test_log_analysis.csv" echo   - test_log_analysis.csv
if exist "sample_metrics.csv" echo   - sample_metrics.csv
echo.
echo 按任意键查看生成的CSV文件...
pause >nul

REM 尝试用默认程序打开CSV文件
if exist "test_log_analysis.csv" start "" "test_log_analysis.csv"
if exist "sample_metrics.csv" start "" "sample_metrics.csv"

echo.
echo 测试完成！
pause
