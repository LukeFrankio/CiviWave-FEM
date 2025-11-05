<#
.SYNOPSIS
    Finds and formats C++ files using clang-format (functional formatting pipeline uwu)

.DESCRIPTION
    This script discovers C++ source files that need formatting according to
    .clang-format style rules and optionally formats them. It excludes external
    dependencies (build/_deps, any vendor/third_party directories).
    
    The script follows functional programming principles:
    - Pure functions for file discovery and filtering
    - Pipeline-based data flow (files flow through transformations)
    - Immutable parameters (using [Parameter()] attributes)
    - Structured return objects (not exceptions for business logic)

.PARAMETER Check
    Check which files need formatting without modifying them.
    Returns exit code 1 if any files need formatting (CI-friendly).

.PARAMETER Fix
    Automatically format files that need formatting.

.PARAMETER Path
    Root path to search for files. Defaults to current directory.

.PARAMETER Verbose
    Enable verbose output showing all checked files.

.EXAMPLE
    .\Format-Code.ps1 -Check
    Checks which files need formatting without modifying them

.EXAMPLE
    .\Format-Code.ps1 -Fix
    Formats all files that need formatting

.EXAMPLE
    .\Format-Code.ps1 -Check -Path "C:\Dev\MyProject" -Verbose
    Checks files in specific directory with verbose output

.NOTES
    Author: LukeFrankio
    Date: 2025-11-05
    Requires: PowerShell 7+, clang-format (latest version preferred!)
    
    This script treats formatting violations as errors because immaculate
    code formatting is self-care uwu ✨

.LINK
    https://clang.llvm.org/docs/ClangFormat.html
#>

[CmdletBinding(DefaultParameterSetName = 'Check')]
param(
    [Parameter(ParameterSetName = 'Check')]
    [switch]$Check,
    
    [Parameter(ParameterSetName = 'Fix')]
    [switch]$Fix,
    
    [Parameter()]
    [ValidateNotNullOrEmpty()]
    [string]$Path = $PSScriptRoot
)

#Requires -Version 7.0

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ============================================================================
# Configuration (constants - immutable after initialization)
# ============================================================================

# File extensions to check (C++ source and header files)
$Script:CppExtensions = @('.cpp', '.hpp', '.cc', '.hh', '.cxx', '.h', '.c')

# Directories to exclude (external dependencies)
$Script:ExcludedDirectories = @(
    '_deps',           # CMake FetchContent dependencies
    'build',           # Build directories
    'vendor',          # Common vendor directory name
    'third_party',     # Common third-party directory name
    'external',        # Common external directory name
    '.git',            # Git metadata
    '.vs',             # Visual Studio temp files
    'CMakeFiles'       # CMake temp files
)

# ============================================================================
# Helper Functions (pure functions where possible)
# ============================================================================

function Test-ClangFormatExists {
    <#
    .SYNOPSIS
        Tests if clang-format is available (pure function!)
    
    .DESCRIPTION
        Checks if clang-format is installed and accessible. This is a PURE function:
        - Same input always produces same output (deterministic)
        - No side effects (doesn't modify anything)
        - Only checks PATH
        
        Returns true if clang-format exists, false otherwise.
    
    .OUTPUTS
        System.Boolean
        True if clang-format exists in PATH, false otherwise
    
    .NOTES
        ✨ PURE FUNCTION ✨
        This is referentially transparent and has no side effects.
    #>
    [CmdletBinding()]
    [OutputType([bool])]
    param()
    
    $null -ne (Get-Command -Name 'clang-format' -ErrorAction SilentlyContinue)
}

function Get-ClangFormatVersion {
    <#
    .SYNOPSIS
        Gets clang-format version string
    
    .DESCRIPTION
        Retrieves version information from clang-format executable.
        Functional approach - returns optional value (string or $null).
    
    .OUTPUTS
        System.String or $null
        Version string if clang-format found, $null otherwise
    
    .NOTES
        ✨ PURE FUNCTION ✨ (mostly - reads external tool version)
        No side effects on system state.
    #>
    [CmdletBinding()]
    [OutputType([string])]
    param()
    
    try {
        $version = & clang-format --version 2>&1 | Out-String
        return $version.Trim()
    }
    catch {
        return $null
    }
}

function Test-PathIsExcluded {
    <#
    .SYNOPSIS
        Tests if path contains excluded directory (pure function!)
    
    .DESCRIPTION
        Checks if file path contains any excluded directories like _deps,
        build, vendor, etc. This is PURE - deterministic and no side effects.
    
    .PARAMETER FilePath
        Path to test
    
    .PARAMETER ExcludedDirs
        Array of directory names to exclude
    
    .OUTPUTS
        System.Boolean
        True if path should be excluded, false otherwise
    
    .EXAMPLE
        Test-PathIsExcluded -FilePath "C:\project\build\_deps\file.cpp"
        True
        # Path contains excluded directory '_deps'
    
    .NOTES
        ✨ PURE FUNCTION ✨
        Referentially transparent, no side effects.
    #>
    [CmdletBinding()]
    [OutputType([bool])]
    param(
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$FilePath,
        
        [Parameter(Mandatory = $true)]
        [string[]]$ExcludedDirs
    )
    
    # normalize path separators for comparison
    $normalizedPath = $FilePath -replace '\\', '/'
    $pathComponents = $normalizedPath -split '/'
    
    # check if any path component matches excluded directories
    foreach ($component in $pathComponents) {
        if ($component -in $ExcludedDirs) {
            return $true
        }
    }
    
    return $false
}

function Get-CppSourceFiles {
    <#
    .SYNOPSIS
        Discovers C++ source files to format (pure discovery function!)
    
    .DESCRIPTION
        Recursively finds all C++ source files in specified path, excluding
        external dependencies and build artifacts. Returns array of FileInfo
        objects for functional pipeline processing.
        
        This is PURE in the sense that it only reads the file system without
        modifying it. Same directory state = same output.
    
    .PARAMETER RootPath
        Root directory to search
    
    .PARAMETER Extensions
        File extensions to include (e.g., .cpp, .hpp)
    
    .PARAMETER ExcludedDirs
        Directory names to exclude from search
    
    .OUTPUTS
        System.IO.FileInfo[]
        Array of FileInfo objects for C++ source files
    
    .EXAMPLE
        $files = Get-CppSourceFiles -RootPath "C:\project" `
                                     -Extensions @('.cpp', '.hpp') `
                                     -ExcludedDirs @('_deps', 'build')
        # Returns all .cpp and .hpp files except those in excluded directories
    
    .NOTES
        ✨ PURE FUNCTION ✨ (read-only file system access)
        No side effects, deterministic for same directory state.
    #>
    [CmdletBinding()]
    [OutputType([System.IO.FileInfo[]])]
    param(
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$RootPath,
        
        [Parameter(Mandatory = $true)]
        [string[]]$Extensions,
        
        [Parameter(Mandatory = $true)]
        [string[]]$ExcludedDirs
    )
    
    Write-Verbose "Searching for C++ files in: $RootPath"
    Write-Verbose "Extensions: $($Extensions -join ', ')"
    Write-Verbose "Excluding directories: $($ExcludedDirs -join ', ')"
    
    # get all files recursively
    $allFiles = Get-ChildItem -Path $RootPath -File -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension -in $Extensions }
    
    # filter out excluded paths (functional pipeline!)
    $filteredFiles = $allFiles | Where-Object {
        -not (Test-PathIsExcluded -FilePath $_.FullName -ExcludedDirs $ExcludedDirs)
    }
    
    # ensure we always return an array (even if empty)
    if ($null -eq $filteredFiles) {
        $filteredFiles = @()
    }
    elseif ($filteredFiles -isnot [array]) {
        $filteredFiles = @($filteredFiles)
    }
    
    Write-Verbose "Found $($filteredFiles.Count) C++ files (excluded external dependencies)"
    
    return $filteredFiles
}

function Test-FileNeedsFormatting {
    <#
    .SYNOPSIS
        Tests if file needs formatting using clang-format
    
    .DESCRIPTION
        Runs clang-format in dry-run mode to check if file would be modified.
        Returns structured result object (functional error handling uwu).
    
    .PARAMETER FilePath
        Path to file to check
    
    .OUTPUTS
        PSCustomObject with properties:
        - FilePath (string): path to file
        - NeedsFormatting (bool): true if file needs formatting
        - Success (bool): true if check succeeded
        - Message (string): status message
    
    .EXAMPLE
        $result = Test-FileNeedsFormatting -FilePath "src/main.cpp"
        if ($result.NeedsFormatting) {
            Write-Host "$($result.FilePath) needs formatting"
        }
    
    .NOTES
        ⚠️ IMPURE FUNCTION (runs external process)
        Has side effects (spawns clang-format process).
    #>
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory = $true, ValueFromPipeline = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$FilePath
    )
    
    process {
        try {
            # run clang-format with dry-run to check if file would be modified
            # --dry-run returns exit code 1 if file would be modified
            $null = & clang-format --dry-run -Werror $FilePath 2>&1
            
            $needsFormatting = $LASTEXITCODE -ne 0
            
            return [PSCustomObject]@{
                FilePath = $FilePath
                NeedsFormatting = $needsFormatting
                Success = $true
                Message = if ($needsFormatting) { 'Needs formatting' } else { 'Already formatted' }
            }
        }
        catch {
            return [PSCustomObject]@{
                FilePath = $FilePath
                NeedsFormatting = $false
                Success = $false
                Message = "Error checking file: $_"
            }
        }
    }
}

function Invoke-FormatFile {
    <#
    .SYNOPSIS
        Formats file using clang-format in-place
    
    .DESCRIPTION
        Runs clang-format with -i flag to format file in-place.
        Returns structured result object (functional error handling).
    
    .PARAMETER FilePath
        Path to file to format
    
    .OUTPUTS
        PSCustomObject with properties:
        - FilePath (string): path to file
        - Success (bool): true if formatting succeeded
        - Message (string): status message
    
    .EXAMPLE
        $result = Invoke-FormatFile -FilePath "src/main.cpp"
        if ($result.Success) {
            Write-Host "Formatted: $($result.FilePath)"
        }
    
    .NOTES
        ⚠️ IMPURE FUNCTION (modifies files)
        Has side effects (modifies file on disk).
    #>
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory = $true, ValueFromPipeline = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$FilePath
    )
    
    process {
        try {
            if ($PSCmdlet.ShouldProcess($FilePath, 'Format file with clang-format')) {
                $output = & clang-format -i $FilePath 2>&1 | Out-String
                
                if ($LASTEXITCODE -eq 0) {
                    return [PSCustomObject]@{
                        FilePath = $FilePath
                        Success = $true
                        Message = 'Formatted successfully'
                    }
                }
                else {
                    return [PSCustomObject]@{
                        FilePath = $FilePath
                        Success = $false
                        Message = "Formatting failed: $output"
                    }
                }
            }
            else {
                return [PSCustomObject]@{
                    FilePath = $FilePath
                    Success = $true
                    Message = 'Skipped (WhatIf mode)'
                }
            }
        }
        catch {
            return [PSCustomObject]@{
                FilePath = $FilePath
                Success = $false
                Message = "Error formatting file: $_"
            }
        }
    }
}

# ============================================================================
# Main Script Logic (functional pipeline!)
# ============================================================================

Write-Host '============================================================' -ForegroundColor Cyan
Write-Host ' clang-format Code Formatting (functional edition uwu)' -ForegroundColor Cyan
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host ''

# Step 1: Check if clang-format exists (pure function)
if (-not (Test-ClangFormatExists)) {
    Write-Error 'clang-format not found in PATH!'
    Write-Host ''
    Write-Host 'Please install clang-format:' -ForegroundColor Yellow
    Write-Host '  - Windows: Install LLVM from https://llvm.org/' -ForegroundColor Yellow
    Write-Host '  - Or install via package manager (choco, scoop, etc.)' -ForegroundColor Yellow
    exit 1
}

$version = Get-ClangFormatVersion
Write-Host "✓ clang-format found: $version" -ForegroundColor Green
Write-Host ''

# Determine mode (Check is default if neither Check nor Fix specified)
$mode = if ($Fix) { 'Fix' } else { 'Check' }
Write-Host "Mode: $mode" -ForegroundColor Cyan
Write-Host "Root path: $Path" -ForegroundColor Gray
Write-Host ''

# Step 2: Discover C++ source files (pure function - just reads file system)
Write-Host 'Discovering C++ source files...' -ForegroundColor Cyan

# determine root path to search (go up from scripts directory to project root if needed)
$searchPath = if (Test-Path (Join-Path $Path 'CMakeLists.txt')) {
    $Path  # already at project root
}
else {
    # assume we're in .github/scripts, go up two levels
    $potentialRoot = Split-Path -Parent (Split-Path -Parent $Path)
    if (Test-Path (Join-Path $potentialRoot 'CMakeLists.txt')) {
        $potentialRoot
    }
    else {
        $Path  # fall back to specified path
    }
}

Write-Verbose "Search path resolved to: $searchPath"

$sourceFiles = Get-CppSourceFiles `
    -RootPath $searchPath `
    -Extensions $Script:CppExtensions `
    -ExcludedDirs $Script:ExcludedDirectories

if ($sourceFiles.Count -eq 0) {
    Write-Host 'No C++ source files found!' -ForegroundColor Yellow
    exit 0
}

Write-Host "✓ Found $($sourceFiles.Count) C++ files (external dependencies excluded)" -ForegroundColor Green
Write-Host ''

# Step 3: Check which files need formatting (functional pipeline!)
Write-Host 'Checking formatting...' -ForegroundColor Cyan

$checkResults = $sourceFiles | ForEach-Object {
    if ($PSCmdlet.MyInvocation.BoundParameters['Verbose']) {
        Write-Host "  Checking: $($_.FullName)" -ForegroundColor Gray
    }
    Test-FileNeedsFormatting -FilePath $_.FullName
}

# ensure results are arrays (even if empty)
$filesToFormat = @($checkResults | Where-Object { $_.NeedsFormatting -and $_.Success })
$checkErrors = @($checkResults | Where-Object { -not $_.Success })

Write-Host ''

# Report check errors if any
if ($checkErrors.Count -gt 0) {
    Write-Host "⚠️ Errors checking $($checkErrors.Count) files:" -ForegroundColor Yellow
    foreach ($errorResult in $checkErrors) {
        Write-Host "  $($errorResult.FilePath): $($errorResult.Message)" -ForegroundColor Yellow
    }
    Write-Host ''
}

# Report formatting results
if ($filesToFormat.Count -eq 0) {
    Write-Host '============================================================' -ForegroundColor Green
    Write-Host ' ✨ All files are properly formatted! ✨' -ForegroundColor Green
    Write-Host '============================================================' -ForegroundColor Green
    Write-Host ''
    Write-Host "Checked $($sourceFiles.Count) files" -ForegroundColor Gray
    Write-Host 'functional formatting gang rise up uwu 💜' -ForegroundColor Magenta
    exit 0
}
else {
    Write-Host "Found $($filesToFormat.Count) files that need formatting:" -ForegroundColor Yellow
    Write-Host ''
    
    foreach ($file in $filesToFormat) {
        $relativePath = Resolve-Path -Relative -Path $file.FilePath
        Write-Host "  - $relativePath" -ForegroundColor Yellow
    }
    
    Write-Host ''
    
    # Step 4: Format files if in Fix mode
    if ($mode -eq 'Fix') {
        Write-Host 'Formatting files...' -ForegroundColor Cyan
        
        $formatResults = $filesToFormat | ForEach-Object {
            $result = Invoke-FormatFile -FilePath $_.FilePath
            if ($result.Success) {
                Write-Host "  ✓ $($result.FilePath)" -ForegroundColor Green
            }
            else {
                Write-Host "  ✗ $($result.FilePath): $($result.Message)" -ForegroundColor Red
            }
            $result
        }
        
        $successCount = ($formatResults | Where-Object { $_.Success }).Count
        $failCount = ($formatResults | Where-Object { -not $_.Success }).Count
        
        Write-Host ''
        Write-Host '============================================================' -ForegroundColor Cyan
        Write-Host " Formatting complete!" -ForegroundColor Cyan
        Write-Host '============================================================' -ForegroundColor Cyan
        Write-Host ''
        Write-Host "Formatted: $successCount files" -ForegroundColor Green
        
        if ($failCount -gt 0) {
            Write-Host "Failed: $failCount files" -ForegroundColor Red
            exit 1
        }
        
        Write-Host ''
        Write-Host 'code formatting goes brrr ✨' -ForegroundColor Magenta
        exit 0
    }
    else {
        # Check mode - report files that need formatting
        Write-Host '============================================================' -ForegroundColor Yellow
        Write-Host ' Files need formatting! Run with -Fix to format.' -ForegroundColor Yellow
        Write-Host '============================================================' -ForegroundColor Yellow
        Write-Host ''
        Write-Host "Files needing formatting: $($filesToFormat.Count)" -ForegroundColor Yellow
        Write-Host "Total files checked: $($sourceFiles.Count)" -ForegroundColor Gray
        Write-Host ''
        Write-Host 'Run: .\Format-Code.ps1 -Fix' -ForegroundColor Cyan
        exit 1
    }
}
