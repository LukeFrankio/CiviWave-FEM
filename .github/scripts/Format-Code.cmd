@echo off
REM Wrapper to run Format-Code.ps1 with PowerShell 7

REM Try pwsh first (if in PATH)
where pwsh >nul 2>&1
if %errorlevel% equ 0 (
    pwsh -ExecutionPolicy Bypass -File "%~dp0Format-Code.ps1" %*
    exit /b %errorlevel%
)

REM Try default PowerShell 7 installation path
if exist "C:\Program Files\PowerShell\7\pwsh.exe" (
    "C:\Program Files\PowerShell\7\pwsh.exe" -ExecutionPolicy Bypass -File "%~dp0Format-Code.ps1" %*
    exit /b %errorlevel%
)

REM PowerShell 7 not found
echo Error: PowerShell 7 is required but not found.
echo Please install PowerShell 7 from: https://github.com/PowerShell/PowerShell/releases
exit /b 1
