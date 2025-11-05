<#
.SYNOPSIS
    Runs clang-tidy on C++ files to check for code quality issues (functional analysis pipeline uwu)

.DESCRIPTION
    This script discovers C++ source files and runs clang-tidy static analysis
    on them. It excludes external dependencies (build/_deps, vendor, third_party)
    from both the file search and the warning/error reports.
    
    The script follows functional programming principles:
    - Pure functions for file discovery and filtering
    - Pipeline-based data flow (files flow through analysis)
    - Immutable parameters (using [Parameter()] attributes)
    - Structured return objects (not exceptions for business logic)

.PARAMETER Path
    Root path to search for files. Defaults to current directory.

.PARAMETER Fix
    Automatically apply clang-tidy fixes where possible (-fix flag).

.PARAMETER BuildPath
    Path to build directory containing compile_commands.json.
    Defaults to .\build

.PARAMETER Checks
    Comma-separated list of checks to enable/disable.
    Defaults to all checks except cert-* and google-* (we prefer our own style).

.PARAMETER Parallel
    Number of parallel clang-tidy processes to run.
    Defaults to number of CPU cores.

.PARAMETER OutputFile
    Path to save the clang-tidy report. If not specified, output goes to console only.

.PARAMETER Verbose
    Enable verbose output showing all checked files.

.EXAMPLE
    .\Run-ClangTidy.ps1
    Runs clang-tidy on all C++ files with default checks

.EXAMPLE
    .\Run-ClangTidy.ps1 -OutputFile "clang_tidy_report.txt"
    Runs clang-tidy and saves report to file

.EXAMPLE
    .\Run-ClangTidy.ps1 -Fix
    Runs clang-tidy and applies automatic fixes

.EXAMPLE
    .\Run-ClangTidy.ps1 -BuildPath "C:\project\build" -Verbose
    Runs with custom build path and verbose output

.EXAMPLE
    .\Run-ClangTidy.ps1 -Checks "modernize-*,readability-*"
    Runs with specific check categories

.NOTES
    Author: LukeFrankio
    Date: 2025-11-05
    Requires: PowerShell 7+, clang-tidy (latest version preferred!), compile_commands.json
    
    This script treats code quality issues as errors because immaculate
    code is self-care uwu ✨

.LINK
    https://clang.llvm.org/extra/clang-tidy/
#>

[CmdletBinding()]
param(
    [Parameter()]
    [ValidateNotNullOrEmpty()]
    [string]$Path = $PSScriptRoot,
    
    [Parameter()]
    [switch]$Fix,
    
    [Parameter()]
    [string]$BuildPath = '',
    
    [Parameter()]
    [string]$Checks = '*,-cert-*,-google-*,-fuchsia-*,-llvm-header-guard,-llvmlibc-*',
    
    [Parameter()]
    [ValidateRange(1, 128)]
    [int]$Parallel = [Environment]::ProcessorCount,
    
    [Parameter()]
    [string]$OutputFile = ''
)

#Requires -Version 7.0

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ============================================================================
# Configuration (constants - immutable after initialization)
# ============================================================================

# File extensions to check (C++ source files - clang-tidy needs .cpp mainly)
$Script:CppExtensions = @('.cpp', '.cc', '.cxx')

# Directories to exclude (external dependencies)
$Script:ExcludedDirectories = @(
    '_deps',           # CMake FetchContent dependencies
    'build',           # Build directories (but we need compile_commands.json from here)
    'vendor',          # Common vendor directory name
    'third_party',     # Common third-party directory name
    'external',        # Common external directory name
    '.git',            # Git metadata
    '.vs',             # Visual Studio temp files
    'CMakeFiles'       # CMake temp files
)

# Patterns to filter warnings from external dependencies
$Script:ExternalDependencyPatterns = @(
    '/_deps/',
    '\\_deps\\',
    '/vendor/',
    '\\vendor\\',
    '/third_party/',
    '\\third_party\\',
    '/external/',
    '\\external\\',
    '/googletest/',
    '\\googletest\\',
    '/googlemock/',
    '\\googlemock\\',
    '/yaml-cpp/',
    '\\yaml-cpp\\',
    '/VulkanMemoryAllocator/',
    '\\VulkanMemoryAllocator\\'
)

# ============================================================================
# Helper Functions (pure functions where possible)
# ============================================================================

function Test-ClangTidyExists {
    <#
    .SYNOPSIS
        Tests if clang-tidy is available (pure function!)
    
    .DESCRIPTION
        Checks if clang-tidy is installed and accessible. This is a PURE function:
        - Same input always produces same output (deterministic)
        - No side effects (doesn't modify anything)
        - Only checks PATH
        
        Returns true if clang-tidy exists, false otherwise.
    
    .OUTPUTS
        System.Boolean
        True if clang-tidy exists in PATH, false otherwise
    
    .NOTES
        ✨ PURE FUNCTION ✨
        This is referentially transparent and has no side effects.
    #>
    [CmdletBinding()]
    [OutputType([bool])]
    param()
    
    $null -ne (Get-Command -Name 'clang-tidy' -ErrorAction SilentlyContinue)
}

function Get-ClangTidyVersion {
    <#
    .SYNOPSIS
        Gets clang-tidy version string
    
    .DESCRIPTION
        Retrieves version information from clang-tidy executable.
        Functional approach - returns optional value (string or $null).
    
    .OUTPUTS
        System.String or $null
        Version string if clang-tidy found, $null otherwise
    
    .NOTES
        ✨ PURE FUNCTION ✨ (mostly - reads external tool version)
        No side effects on system state.
    #>
    [CmdletBinding()]
    [OutputType([string])]
    param()
    
    try {
        $version = & clang-tidy --version 2>&1 | Out-String
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
    # (but allow reading compile_commands.json from build directory)
    foreach ($component in $pathComponents) {
        if ($component -in $ExcludedDirs -and $component -ne 'build') {
            return $true
        }
    }
    
    return $false
}

function Test-LineIsExternalDependency {
    <#
    .SYNOPSIS
        Tests if output line references external dependency (pure function!)
    
    .DESCRIPTION
        Checks if clang-tidy output line contains path to external dependency.
        This allows filtering out warnings from third-party code.
    
    .PARAMETER Line
        Output line to test
    
    .PARAMETER ExternalPatterns
        Array of path patterns that indicate external dependencies
    
    .OUTPUTS
        System.Boolean
        True if line references external dependency, false otherwise
    
    .NOTES
        ✨ PURE FUNCTION ✨
        Referentially transparent, no side effects.
    #>
    [CmdletBinding()]
    [OutputType([bool])]
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string]$Line,
        
        [Parameter(Mandatory = $true)]
        [string[]]$ExternalPatterns
    )
    
    foreach ($pattern in $ExternalPatterns) {
        if ($Line -like "*$pattern*") {
            return $true
        }
    }
    
    return $false
}

function Get-CppSourceFiles {
    <#
    .SYNOPSIS
        Discovers C++ source files to analyze (pure discovery function!)
    
    .DESCRIPTION
        Recursively finds all C++ source files in specified path, excluding
        external dependencies and build artifacts. Returns array of FileInfo
        objects for functional pipeline processing.
        
        This is PURE in the sense that it only reads the file system without
        modifying it. Same directory state = same output.
    
    .PARAMETER RootPath
        Root directory to search
    
    .PARAMETER Extensions
        File extensions to include (e.g., .cpp, .cc)
    
    .PARAMETER ExcludedDirs
        Directory names to exclude from search
    
    .OUTPUTS
        System.IO.FileInfo[]
        Array of FileInfo objects for C++ source files
    
    .EXAMPLE
        $files = Get-CppSourceFiles -RootPath "C:\project" `
                                     -Extensions @('.cpp', '.cc') `
                                     -ExcludedDirs @('_deps', 'vendor')
        # Returns all .cpp and .cc files except those in excluded directories
    
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

function Find-CompileCommandsJson {
    <#
    .SYNOPSIS
        Finds compile_commands.json in build directory
    
    .DESCRIPTION
        Searches for compile_commands.json which is required for clang-tidy
        to understand the project's compilation settings.
    
    .PARAMETER BuildPath
        Path to build directory (if specified) or project root
    
    .OUTPUTS
        System.String or $null
        Path to compile_commands.json if found, $null otherwise
    
    .NOTES
        ✨ PURE FUNCTION ✨ (read-only file system access)
        No side effects.
    #>
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Mandatory = $true)]
        [string]$BuildPath
    )
    
    $compileCommandsPath = Join-Path $BuildPath 'compile_commands.json'
    
    if (Test-Path -Path $compileCommandsPath -PathType Leaf) {
        return $compileCommandsPath
    }
    
    return $null
}

function New-FilteredCompileCommands {
    <#
    .SYNOPSIS
        Creates filtered compile_commands.json for clang-tidy compatibility
    
    .DESCRIPTION
        Reads compile_commands.json and removes C++20 module-specific compiler
        flags that clang-tidy doesn't understand. These flags include:
        - -fmodules-ts (C++20 modules)
        - -fmodule-mapper=... (module mapping files)
        - -fdeps-format=p1689r5 (dependency format)
        - -MD (dependency generation when used with module flags)
        - -x c++ (explicit language specification conflicts with clang-tidy)
        
        This is necessary because GCC's module implementation uses flags that
        clang-tidy (based on clang) doesn't recognize.
    
    .PARAMETER SourcePath
        Path to original compile_commands.json
    
    .PARAMETER DestinationPath
        Path where filtered compile_commands.json will be written
    
    .OUTPUTS
        PSCustomObject with properties:
        - Success (bool): true if filtering succeeded
        - Message (string): status message
        - FilteredPath (string): path to filtered file
    
    .NOTES
        ⚠️ IMPURE FUNCTION (reads and writes files)
        Creates temporary filtered compile_commands.json for clang-tidy.
    #>
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$SourcePath,
        
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$DestinationPath
    )
    
    try {
        Write-Verbose "Filtering compile_commands.json for clang-tidy compatibility..."
        
        # read original compile_commands.json
        $compileCommandsContent = Get-Content -Path $SourcePath -Raw -Encoding UTF8
        $compileCommands = $compileCommandsContent | ConvertFrom-Json
        
        # filter out problematic compiler flags (functional pipeline!)
        $filteredCommands = $compileCommands | ForEach-Object {
            $entry = $_
            
            # remove C++20 module flags that clang-tidy doesn't understand
            # these are GCC-specific module implementation flags
            $filteredCommand = $entry.command `
                -replace '-fmodules-ts\s*', '' `
                -replace '-fmodule-mapper=[^\s]+\s*', '' `
                -replace '-fdeps-format=[^\s]+\s*', '' `
                -replace '-MD\s+', '' `
                -replace '-x\s+c\+\+\s+', ''
            
            # create new entry with filtered command
            [PSCustomObject]@{
                directory = $entry.directory
                command = $filteredCommand.Trim()
                file = $entry.file
                output = $entry.output
            }
        }
        
        # write filtered compile_commands.json
        $filteredJson = $filteredCommands | ConvertTo-Json -Depth 10
        $filteredJson | Out-File -FilePath $DestinationPath -Encoding UTF8 -Force
        
        Write-Verbose "Created filtered compile_commands.json: $DestinationPath"
        
        return [PSCustomObject]@{
            Success = $true
            Message = "Successfully created filtered compile_commands.json"
            FilteredPath = $DestinationPath
        }
    }
    catch {
        return [PSCustomObject]@{
            Success = $false
            Message = "Failed to filter compile_commands.json: $_"
            FilteredPath = $null
        }
    }
}

function Invoke-ClangTidyOnFile {
    <#
    .SYNOPSIS
        Runs clang-tidy on a single file
    
    .DESCRIPTION
        Executes clang-tidy analysis on specified file and returns structured
        result with filtered output (excluding external dependency warnings).
    
    .PARAMETER FilePath
        Path to file to analyze
    
    .PARAMETER CompileCommandsDir
        Directory containing compile_commands.json
    
    .PARAMETER Checks
        Clang-tidy checks to enable
    
    .PARAMETER Fix
        Whether to apply automatic fixes
    
    .PARAMETER ExternalPatterns
        Patterns to filter external dependency warnings
    
    .OUTPUTS
        PSCustomObject with properties:
        - FilePath (string): path to analyzed file
        - Success (bool): true if analysis succeeded
        - HasIssues (bool): true if issues found (after filtering)
        - Output (string): filtered clang-tidy output
        - RawOutput (string): unfiltered output for debugging
    
    .NOTES
        ⚠️ IMPURE FUNCTION (runs external process)
        Has side effects (spawns clang-tidy, may modify file if -Fix used).
    #>
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory = $true, ValueFromPipeline = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$FilePath,
        
        [Parameter(Mandatory = $true)]
        [string]$CompileCommandsDir,
        
        [Parameter(Mandatory = $true)]
        [string]$Checks,
        
        [Parameter()]
        [switch]$Fix,
        
        [Parameter(Mandatory = $true)]
        [string[]]$ExternalPatterns
    )
    
    process {
        try {
            # build clang-tidy arguments
            $tidyArgs = @(
                $FilePath
                "--checks=$Checks"
                "-p=$CompileCommandsDir"
            )
            
            if ($Fix) {
                $tidyArgs += '--fix'
            }
            
            # run clang-tidy and capture output
            $output = & clang-tidy @tidyArgs 2>&1 | Out-String
            
            # filter out warnings from external dependencies (functional filtering!)
            $lines = $output -split "`n"
            $filteredLines = $lines | Where-Object {
                -not (Test-LineIsExternalDependency -Line $_ -ExternalPatterns $ExternalPatterns)
            }
            
            $filteredOutput = $filteredLines -join "`n"
            
            # check if there are actual issues (warnings/errors) in our code
            $hasIssues = $filteredOutput -match '(warning:|error:)' -and 
                         $filteredOutput -notmatch 'no errors found'
            
            return [PSCustomObject]@{
                FilePath = $FilePath
                Success = $true
                HasIssues = $hasIssues
                Output = $filteredOutput.Trim()
                RawOutput = $output
            }
        }
        catch {
            return [PSCustomObject]@{
                FilePath = $FilePath
                Success = $false
                HasIssues = $false
                Output = "Error running clang-tidy: $_"
                RawOutput = ""
            }
        }
    }
}

# ============================================================================
# Main Script Logic (functional pipeline!)
# ============================================================================

Write-Host '============================================================' -ForegroundColor Cyan
Write-Host ' clang-tidy Static Analysis (functional edition uwu)' -ForegroundColor Cyan
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host ''

# Step 1: Resolve build path
if (-not $BuildPath) {
    # default to ./build relative to current directory
    $BuildPath = Join-Path $PWD 'build'
}

# Convert to absolute path
$BuildPath = Resolve-Path -Path $BuildPath -ErrorAction SilentlyContinue
if (-not $BuildPath) {
    Write-Error "Build directory not found. Please specify -BuildPath or ensure ./build exists."
    exit 1
}

# Step 2: Check if clang-tidy exists (pure function)
if (-not (Test-ClangTidyExists)) {
    Write-Error 'clang-tidy not found in PATH!'
    Write-Host ''
    Write-Host 'Please install clang-tidy:' -ForegroundColor Yellow
    Write-Host '  - Windows: Install LLVM from https://llvm.org/' -ForegroundColor Yellow
    Write-Host '  - Or install via package manager (choco, scoop, etc.)' -ForegroundColor Yellow
    exit 1
}

$version = Get-ClangTidyVersion
Write-Host "✓ clang-tidy found: $version" -ForegroundColor Green
Write-Host ''

# Step 3: Find compile_commands.json
$compileCommands = Find-CompileCommandsJson -BuildPath $BuildPath

if (-not $compileCommands) {
    Write-Error "compile_commands.json not found in build directory: $BuildPath"
    Write-Host ''
    Write-Host 'Please run CMake with CMAKE_EXPORT_COMPILE_COMMANDS=ON:' -ForegroundColor Yellow
    Write-Host '  cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON' -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ Found compile_commands.json: $compileCommands" -ForegroundColor Green
Write-Host ''

# Step 3.5: Create filtered compile_commands.json for clang-tidy
# (removes C++20 module flags that clang-tidy doesn't understand)
$filteredCompileCommands = Join-Path $BuildPath 'compile_commands_filtered.json'
$filterResult = New-FilteredCompileCommands -SourcePath $compileCommands -DestinationPath $filteredCompileCommands

if (-not $filterResult.Success) {
    Write-Error $filterResult.Message
    exit 1
}

Write-Host "✓ Created filtered compile_commands.json (removed module flags)" -ForegroundColor Green
Write-Host "  Filtered: $filteredCompileCommands" -ForegroundColor Gray
Write-Host ''

# use filtered compile commands for clang-tidy
$compileCommandsForTidy = $filteredCompileCommands

# Step 4: Configuration summary
Write-Host 'Configuration:' -ForegroundColor Cyan
Write-Host "  Root path: $Path" -ForegroundColor Gray
Write-Host "  Build path: $BuildPath" -ForegroundColor Gray
Write-Host "  Compile commands: $compileCommandsForTidy" -ForegroundColor Gray
Write-Host "  Checks: $Checks" -ForegroundColor Gray
Write-Host "  Parallel jobs: $Parallel" -ForegroundColor Gray
Write-Host "  Fix mode: $(if ($Fix) { 'Enabled' } else { 'Disabled' })" -ForegroundColor Gray
Write-Host ''

# Step 5: Discover C++ source files (pure function - just reads file system)
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

# Step 6: Run clang-tidy on files (functional pipeline with parallelization!)
Write-Host 'Running clang-tidy analysis...' -ForegroundColor Cyan
Write-Host "  (Using $Parallel parallel jobs)" -ForegroundColor Gray
Write-Host ''

# run clang-tidy in parallel using ForEach-Object -Parallel (PowerShell 7 feature!)
$results = $sourceFiles | ForEach-Object -Parallel {
    # get parameters from parent scope
    $file = $_
    $compileCommandsDir = Split-Path -Parent ${using:compileCommandsForTidy}
    $checks = ${using:Checks}
    $fix = ${using:Fix}
    $externalPatterns = ${using:Script:ExternalDependencyPatterns}
    $verbose = ${using:VerbosePreference}
    
    if ($verbose -eq 'Continue') {
        Write-Host "  Analyzing: $($file.FullName)" -ForegroundColor Gray
    }
    
    # build clang-tidy arguments
    $tidyArgs = @(
        $file.FullName
        "--checks=$checks"
        "-p=$compileCommandsDir"
    )
    
    if ($fix) {
        $tidyArgs += '--fix'
    }
    
    try {
        # run clang-tidy and capture output
        $output = & clang-tidy @tidyArgs 2>&1 | Out-String
        
        # filter out warnings from external dependencies
        $lines = $output -split "`n"
        $filteredLines = $lines | Where-Object {
            $line = $_
            $isExternal = $false
            foreach ($pattern in $externalPatterns) {
                if ($line -like "*$pattern*") {
                    $isExternal = $true
                    break
                }
            }
            -not $isExternal
        }
        
        $filteredOutput = $filteredLines -join "`n"
        
        # check if there are actual issues (warnings/errors) in our code
        $hasIssues = $filteredOutput -match '(warning:|error:)' -and 
                     $filteredOutput -notmatch 'no errors found'
        
        [PSCustomObject]@{
            FilePath = $file.FullName
            Success = $true
            HasIssues = $hasIssues
            Output = $filteredOutput.Trim()
            RawOutput = $output
        }
    }
    catch {
        [PSCustomObject]@{
            FilePath = $file.FullName
            Success = $false
            HasIssues = $false
            Output = "Error running clang-tidy: $_"
            RawOutput = ""
        }
    }
} -ThrottleLimit $Parallel

Write-Host ''

# Step 7: Process results (functional aggregation!)
# ensure results are arrays (even if empty)
$filesWithIssues = @($results | Where-Object { $_.HasIssues })
$failedAnalysis = @($results | Where-Object { -not $_.Success })
$cleanFiles = @($results | Where-Object { $_.Success -and -not $_.HasIssues })

# Report analysis failures
if ($failedAnalysis.Count -gt 0) {
    Write-Host "⚠️ Failed to analyze $($failedAnalysis.Count) files:" -ForegroundColor Yellow
    foreach ($failure in $failedAnalysis) {
        Write-Host "  $($failure.FilePath)" -ForegroundColor Yellow
        Write-Host "    $($failure.Output)" -ForegroundColor Gray
    }
    Write-Host ''
}

# Report issues found
if ($filesWithIssues.Count -eq 0) {
    $successMessage = @"
============================================================
 ✨ No code quality issues found! ✨
============================================================

Analyzed $($sourceFiles.Count) files
Clean files: $($cleanFiles.Count)

functional code quality gang rise up uwu 💜
"@
    
    Write-Host $successMessage -ForegroundColor Green
    
    # Save to file if requested
    if ($OutputFile) {
        $successMessage | Out-File -FilePath $OutputFile -Encoding UTF8
        Write-Host "Report saved to: $OutputFile" -ForegroundColor Cyan
    }
    
    exit 0
}
else {
    # Build report content
    $reportLines = @()
    $reportLines += '============================================================'
    $reportLines += " Found issues in $($filesWithIssues.Count) files"
    $reportLines += '============================================================'
    $reportLines += ''
    
    # output detailed issues for each file
    foreach ($fileResult in $filesWithIssues) {
        $relativePath = Resolve-Path -Relative -Path $fileResult.FilePath
        Write-Host "File: $relativePath" -ForegroundColor Cyan
        Write-Host $fileResult.Output -ForegroundColor Gray
        Write-Host ''
        
        # Add to report
        $reportLines += "File: $relativePath"
        $reportLines += $fileResult.Output
        $reportLines += ''
    }
    
    $reportLines += '============================================================'
    $reportLines += ' Summary'
    $reportLines += '============================================================'
    $reportLines += "Total files analyzed: $($sourceFiles.Count)"
    $reportLines += "Files with issues: $($filesWithIssues.Count)"
    $reportLines += "Clean files: $($cleanFiles.Count)"
    
    if ($failedAnalysis.Count -gt 0) {
        Write-Host "Failed to analyze: $($failedAnalysis.Count)" -ForegroundColor Red
        $reportLines += "Failed to analyze: $($failedAnalysis.Count)"
    }
    
    Write-Host ''
    $reportLines += ''
    
    if ($Fix) {
        Write-Host 'Automatic fixes applied where possible.' -ForegroundColor Cyan
        Write-Host 'Re-run without -Fix to verify remaining issues.' -ForegroundColor Cyan
        $reportLines += 'Automatic fixes applied where possible.'
        $reportLines += 'Re-run without -Fix to verify remaining issues.'
    }
    else {
        Write-Host 'Run with -Fix to automatically apply fixes.' -ForegroundColor Cyan
        $reportLines += 'Run with -Fix to automatically apply fixes.'
    }
    
    # Save report to file if requested
    if ($OutputFile) {
        $reportContent = $reportLines -join "`n"
        $reportContent | Out-File -FilePath $OutputFile -Encoding UTF8
        Write-Host ''
        Write-Host "Report saved to: $OutputFile" -ForegroundColor Cyan
    }
    
    exit 1
}
