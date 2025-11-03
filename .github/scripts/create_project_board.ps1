<#
Creates a GitHub project board (classic) for a repository and adds columns + milestones.

Usage (PowerShell):
  $env:GITHUB_TOKEN = '<YOUR_TOKEN>'
  .\.github\scripts\create_project_board.ps1 -Owner 'LukeFrankio' -Repo 'CiviWave-FEM' -ProjectName 'CiviWave Board'

Notes:
- Requires a Personal Access Token or Fine-grained token with repo administration scope.
- If `gh` (GitHub CLI) is available the script will prefer it for project creation; otherwise it uses the REST API via Invoke-RestMethod.
- The script creates columns: Backlog, In Progress, Review, Done and sample milestones M1..M6.
- Replace or extend the `$Columns` and `$Milestones` arrays below to customize.
#>

param(
    [string]$Owner = 'LukeFrankio',
    [string]$Repo = 'CiviWave-FEM',
    [string]$ProjectName = 'CiviWave-FEM Board',
    [string]$ProjectBody = 'Project board generated from repository scaffolding assets.',
    [switch]$UseClassic = $true
)

function Require-Token {
    if (-not $env:GITHUB_TOKEN) {
        Write-Error "Please set the environment variable GITHUB_TOKEN to a token that has repo administration permissions (or use gh auth login and ensure gh is available)."
        exit 2
    }
}

$Columns = @('Backlog','In Progress','Review','Done')
$Milestones = @('M1','M2','M3','M4','M5','M6')

# Prefer gh CLI if installed and authenticated
if (Get-Command gh -ErrorAction SilentlyContinue) {
    try {
        # Test authentication
        gh auth status -h github.com > $null 2>&1
        $ghAvailable = $true
    } catch {
        $ghAvailable = $false
    }
} else {
    $ghAvailable = $false
}

if ($ghAvailable) {
    Write-Host "gh CLI detected and authenticated. Creating project via gh..."
    # Use the gh CLI directly; avoid Bash-style line continuations and ensure
    # PowerShell treats -f as an argument by quoting the whole key=value pair.
    try {
        $projCreate = & gh api --method POST -H 'Accept: application/vnd.github+json' "/repos/$Owner/$Repo/projects" -f "name=$ProjectName" -f "body=$ProjectBody" 2>&1
        $proj = $projCreate | ConvertFrom-Json
    } catch {
        Write-Error "Failed to create project via gh: $projCreate`nException: $_"
        exit 3
    }
    if (-not $proj.id) {
        Write-Error "Failed to create project via gh: $projCreate"
        exit 3
    }
    $project_id = $proj.id
    Write-Host "Created project id: $project_id"
} else {
    Require-Token
    $Headers = @{ Authorization = "Bearer $($env:GITHUB_TOKEN)"; Accept = 'application/vnd.github+json'; 'X-GitHub-Api-Version' = '2022-11-28' }

    $projectPayload = @{ name = $ProjectName; body = $ProjectBody; private = $false } | ConvertTo-Json
    Write-Host "Creating project via REST API..."
    try {
        $proj = Invoke-RestMethod -Uri "https://api.github.com/repos/$Owner/$Repo/projects" -Method POST -Headers $Headers -Body $projectPayload -ContentType 'application/json'
    } catch {
        Write-Error "Project creation failed. Check token permissions and that Projects (classic) API is enabled for your account. Error: $_"
        exit 4
    }
    $project_id = $proj.id
    Write-Host "Created project id: $project_id"
}

# Create columns
if ($project_id) {
    foreach ($col in $Columns) {
        $colPayload = @{ name = $col } | ConvertTo-Json
        if ($ghAvailable) {
            try {
                $colRespRaw = & gh api --method POST -H 'Accept: application/vnd.github+json' "/projects/$project_id/columns" -f "name=$col" 2>&1
                $colResp = $colRespRaw | ConvertFrom-Json
                Write-Host "Created column: $($colResp.name) (id:$($colResp.id))"
            } catch {
                Write-Warning "Failed to create column '$col' via gh: $_"
            }
        } else {
            try {
                $colResp = Invoke-RestMethod -Uri "https://api.github.com/projects/$project_id/columns" -Method POST -Headers $Headers -Body $colPayload -ContentType 'application/json'
                Write-Host "Created column: $($colResp.name) (id:$($colResp.id))"
            } catch {
                Write-Warning "Failed to create column '$col': $_"
            }
        }
    }
}

# Create milestones in the repository
foreach ($m in $Milestones) {
    $description = "Milestone $m for phased work"
    if ($ghAvailable) {
        try {
            $msRaw = & gh api --method POST -H 'Accept: application/vnd.github+json' "/repos/$Owner/$Repo/milestones" -f "title=$m" -f "state=open" -f "description=$description" 2>&1
            $ms = $msRaw | ConvertFrom-Json
            Write-Host "Created milestone: $($ms.title) (number: $($ms.number))"
        } catch {
            Write-Warning "Failed to create milestone '$m' via gh: $_"
        }
    } else {
        $msPayload = @{ title = $m; state = 'open'; description = $description } | ConvertTo-Json
        try {
            $ms = Invoke-RestMethod -Uri "https://api.github.com/repos/$Owner/$Repo/milestones" -Method POST -Headers $Headers -Body $msPayload -ContentType 'application/json'
            Write-Host "Created milestone: $($ms.title) (number: $($ms.number))"
        } catch {
            Write-Warning "Failed to create milestone '$m': $_"
        }
    }
}

Write-Host "Project board creation finished. Open the repository Projects tab to view the result."

Write-Host "Notes: If you prefer Projects (v2) or GitHub Projects in the new UI, you may need to use the GitHub web UI or GraphQL API; this script targets Projects (classic) via the REST endpoints."