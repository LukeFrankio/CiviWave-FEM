<#
.SYNOPSIS
    Captures RGP profile for CiviWave-FEM simulation workloads uwu

.DESCRIPTION
    This script automates AMD Radeon GPU Profiler (RGP) capture for CiviWave-FEM.
    It launches the application with RGP injection enabled, waits for warmup frames,
    then triggers capture via RGP's command-line interface.

    Prerequisites:
    - AMD Radeon GPU Profiler installed (part of AMD GPUOpen tools)
    - RGP CLI (rgp.exe or RadeonGPUProfiler.exe) on PATH or RGP_PATH env var set
    - CiviWave-FEM built with ENABLE_RGP_MARKERS=ON

.PARAMETER Executable
    Path to the CiviWave-FEM executable to profile.
    Defaults to build/bin/cwf_viewer_demo.exe

.PARAMETER OutputDir
    Directory where RGP captures will be saved.
    Defaults to ./rgp_captures

.PARAMETER WarmupFrames
    Number of frames to skip before starting capture (allows steady state).
    Default: 60

.PARAMETER CaptureFrames
    Number of frames to capture.
    Default: 10

.PARAMETER ScenarioFile
    Optional YAML scenario file to pass to the application.

.EXAMPLE
    .\Capture-Rgp.ps1 -WarmupFrames 120 -CaptureFrames 20

.NOTES
    Author: LukeFrankio
    Date: 2025-11-25
    Version: 1.0
#>
[CmdletBinding()]
param(
    [Parameter()]
    [string]$Executable = "build\bin\cwf_viewer_demo.exe",

    [Parameter()]
    [string]$OutputDir = "rgp_captures",

    [Parameter()]
    [int]$WarmupFrames = 60,

    [Parameter()]
    [int]$CaptureFrames = 10,

    [Parameter()]
    [string]$ScenarioFile = ""
)

#Requires -Version 7.0

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

function Write-Status {
    param([string]$Message, [string]$Color = "Cyan")
    Write-Host "[cwf::rgp] " -ForegroundColor DarkGray -NoNewline
    Write-Host $Message -ForegroundColor $Color
}

function Test-RgpAvailable {
    <#
    .SYNOPSIS
        Checks if RGP CLI is available on the system
    #>

    # Check RGP_PATH environment variable first
    if ($env:RGP_PATH -and (Test-Path $env:RGP_PATH)) {
        return $env:RGP_PATH
    }

    # Common installation paths
    $commonPaths = @(
        "C:\Program Files\AMD\RadeonGPUProfiler\RadeonGPUProfiler.exe",
        "C:\Program Files (x86)\AMD\RadeonGPUProfiler\RadeonGPUProfiler.exe",
        "${env:ProgramFiles}\AMD\RadeonGPUProfiler\RadeonGPUProfiler.exe",
        "${env:LOCALAPPDATA}\RadeonGPUProfiler\RadeonGPUProfiler.exe"
    )

    foreach ($path in $commonPaths) {
        if (Test-Path $path) {
            return $path
        }
    }

    # Check if rgp is on PATH
    $rgpOnPath = Get-Command "RadeonGPUProfiler.exe" -ErrorAction SilentlyContinue
    if ($rgpOnPath) {
        return $rgpOnPath.Source
    }

    return $null
}

function Get-TimestampedFilename {
    <#
    .SYNOPSIS
        Generates a timestamped filename for RGP captures
    #>
    param([string]$Prefix = "cwf_capture")

    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    return "${Prefix}_${timestamp}.rgp"
}

# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------

Write-Status "CiviWave-FEM RGP Capture Script" "Magenta"
Write-Status "================================" "Magenta"

# Validate executable exists
$execPath = Resolve-Path $Executable -ErrorAction SilentlyContinue
if (-not $execPath) {
    Write-Status "ERROR: Executable not found: $Executable" "Red"
    Write-Status "Build the project first with BUILD_UI=ON" "Yellow"
    exit 1
}
Write-Status "Executable: $execPath" "Green"

# Find RGP installation
$rgpPath = Test-RgpAvailable
if (-not $rgpPath) {
    Write-Status "ERROR: AMD Radeon GPU Profiler not found" "Red"
    Write-Status "Install RGP from: https://gpuopen.com/rgp/" "Yellow"
    Write-Status "Or set RGP_PATH environment variable" "Yellow"
    exit 1
}
Write-Status "RGP found: $rgpPath" "Green"

# Create output directory
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    Write-Status "Created output directory: $OutputDir" "Green"
}

# Generate capture filename
$captureFilename = Get-TimestampedFilename
$capturePath = Join-Path $OutputDir $captureFilename
Write-Status "Capture output: $capturePath" "Green"

# Build command-line arguments
$appArgs = @()
if ($ScenarioFile -and (Test-Path $ScenarioFile)) {
    $appArgs += "--scenario"
    $appArgs += $ScenarioFile
    Write-Status "Scenario: $ScenarioFile" "Green"
}

# Configure RGP environment variables for capture
$env:AMD_RGP_CAPTURE_ON_PRESENT = "1"
$env:AMD_RGP_WARMUP_FRAMES = $WarmupFrames.ToString()
$env:AMD_RGP_CAPTURE_FRAMES = $CaptureFrames.ToString()
$env:AMD_RGP_OUTPUT_FILE = $capturePath

Write-Status "Configuration:" "Yellow"
Write-Status "  Warmup frames: $WarmupFrames" "Gray"
Write-Status "  Capture frames: $CaptureFrames" "Gray"
Write-Status "" ""

# Launch application with RGP injection
Write-Status "Launching application with RGP injection..." "Cyan"
Write-Status "Press Ctrl+Shift+C during execution to trigger manual capture" "Yellow"
Write-Status "" ""

try {
    # Start the application process
    $processInfo = Start-Process -FilePath $execPath -ArgumentList $appArgs -PassThru -NoNewWindow

    Write-Status "Application PID: $($processInfo.Id)" "Green"
    Write-Status "Waiting for capture to complete..." "Cyan"
    Write-Status "(Close the application window when done)" "Gray"

    # Wait for the process to exit
    $processInfo.WaitForExit()

    $exitCode = $processInfo.ExitCode
    if ($exitCode -eq 0) {
        Write-Status "Application exited cleanly" "Green"
    }
    else {
        Write-Status "Application exited with code: $exitCode" "Yellow"
    }
}
catch {
    Write-Status "ERROR: Failed to launch application: $_" "Red"
    exit 1
}

# Check if capture file was created
if (Test-Path $capturePath) {
    $fileInfo = Get-Item $capturePath
    $sizeMB = [math]::Round($fileInfo.Length / 1MB, 2)
    Write-Status "SUCCESS: Capture saved ($sizeMB MB)" "Green"
    Write-Status "  Path: $capturePath" "Gray"
    Write-Status "" ""
    Write-Status "Open in RGP:" "Cyan"
    Write-Status "  & '$rgpPath' '$capturePath'" "White"
}
else {
    Write-Status "WARNING: No capture file found at expected location" "Yellow"
    Write-Status "Capture may have failed or been saved elsewhere" "Yellow"

    # Check for any .rgp files in output directory
    $rgpFiles = Get-ChildItem -Path $OutputDir -Filter "*.rgp" -ErrorAction SilentlyContinue
    if ($rgpFiles) {
        Write-Status "Found RGP files in output directory:" "Cyan"
        foreach ($file in $rgpFiles) {
            Write-Status "  $($file.FullName)" "Gray"
        }
    }
}

Write-Status "" ""
Write-Status "Done! uwu" "Magenta"
