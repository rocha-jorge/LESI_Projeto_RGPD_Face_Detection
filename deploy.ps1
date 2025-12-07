<#
Windows installer script for face-pipeline
Usage (run in an elevated PowerShell prompt):
  .\deploy.ps1

This script copies repository files to `C:\srv\face-pipeline`, creates required
folders and either builds/loads the Docker image (if Docker is present) or sets
up a Python venv and installs dependencies as a fallback.
#>

param(
    [string] $InstallDir = "C:\srv\face-pipeline",
    [string] $TarballPath = "",
    [switch] $SkipDocker
)

Set-StrictMode -Version Latest

function Require-Admin {
    if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
        Write-Error "This script must be run as Administrator. Re-run PowerShell as Administrator and try again."
        exit 1
    }
}

Require-Admin

Write-Host "Installing face-pipeline to $InstallDir"

# Create install dir and copy files
if (-Not (Test-Path $InstallDir)) { New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null }

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Copying files to $InstallDir (excluding venv and large folders)..."
$excludes = @("venv_face_detection","venv","photo_input","photo_detection_output","photo_detection_error","photo_anonymization_output","weights",".git")
Get-ChildItem -Path $scriptDir -Force | Where-Object { $excludes -notcontains $_.Name } | ForEach-Object {
    $dest = Join-Path $InstallDir $_.Name
    if ($_.PSIsContainer) {
        Copy-Item -Path $_.FullName -Destination $dest -Recurse -Force
    } else {
        Copy-Item -Path $_.FullName -Destination $dest -Force
    }
}

# Ensure host folders exist
$folders = @("photo_input","photo_detection_output","photo_detection_error","photo_anonymization_output","models")
foreach ($f in $folders) {
    $p = Join-Path $InstallDir $f
    if (-Not (Test-Path $p)) { New-Item -ItemType Directory -Path $p -Force | Out-Null }
}

# If Docker is available and not skipped, build or load image
$dockerAvailable = (Get-Command docker -ErrorAction SilentlyContinue) -ne $null
if ($dockerAvailable -and -not $SkipDocker) {
    Push-Location $InstallDir
    try {
        if ($TarballPath -and (Test-Path $TarballPath)) {
            Write-Host "Loading provided image tarball: $TarballPath"
            docker load -i $TarballPath
        } else {
            Write-Host "Building docker image (this may take a while)..."
            docker build -t face-pipeline:latest .
        }
        Write-Host "Starting container via docker compose..."
        if (Get-Command "docker-compose" -ErrorAction SilentlyContinue) {
            docker-compose up -d
        } else {
            docker compose up -d
        }
    } catch {
        Write-Error "Docker build/run failed: $_"
    } finally {
        Pop-Location
    }
} else {
    Write-Host "Docker not available or skipped. Setting up Python venv fallback..."
    $venvPath = Join-Path $InstallDir "venv"
    python -m venv $venvPath
    $activate = Join-Path $venvPath "Scripts\Activate.ps1"
    & $activate
    pip install -r (Join-Path $InstallDir "requirements.txt")
    Write-Host "Created venv and installed requirements. To run the watcher: `& $venvPath\Scripts\python.exe src\watcher.py`"
}

Write-Host "Installation finished."
Write-Host "If you want a Windows service wrapper, use NSSM (see deploy/NSSM_INSTRUCTIONS.md) or create a Scheduled Task."
