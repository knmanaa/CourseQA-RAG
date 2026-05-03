# Run from the repo root:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_conda_env.ps1 [EnvName] [LlamaCppVersion]

param(
    [string]$EnvName       = "CourseQARAG",
    [string]$LlamaCppVersion = "0.3.20"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir
$YmlFile   = Join-Path $RepoRoot "conda_env_setup.yml"
$ReqFile   = Join-Path $RepoRoot "requirements.txt"

# ---------------------------------------------------------------------------
# Helper: import env vars from vcvarsall.bat into the current PS session.
# Uses regex so values containing '=' are handled correctly (unlike bash IFS split).
# ---------------------------------------------------------------------------
function Import-VcVarsAll {
    param([string]$VcVarsAllPath)
    $output = cmd /c "`"$VcVarsAllPath`" x64 && set" 2>&1
    foreach ($line in $output) {
        if ($line -match '^([^=]+)=(.*)$') {
            [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
        }
    }
}

# ---------------------------------------------------------------------------
# Helper: locate conda executable
# ---------------------------------------------------------------------------
function Find-Conda {
    # Already on PATH?
    $c = Get-Command conda -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }

    $candidates = @(
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:USERPROFILE\Miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\Anaconda3\Scripts\conda.exe",
        "C:\ProgramData\miniconda3\Scripts\conda.exe",
        "C:\ProgramData\anaconda3\Scripts\conda.exe",
        "C:\tools\miniconda3\Scripts\conda.exe"
    )
    foreach ($p in $candidates) {
        if (Test-Path $p) { return $p }
    }
    return $null
}

# ---------------------------------------------------------------------------
# Helper: get the Python exe path for an env by name
# ---------------------------------------------------------------------------
function Get-CondaEnvPython {
    param([string]$CondaExe, [string]$Name)
    $json = & $CondaExe info --json 2>$null | ConvertFrom-Json
    foreach ($p in $json.envs) {
        if ((Split-Path -Leaf $p) -eq $Name) {
            $win = Join-Path $p "python.exe"
            if (Test-Path $win) { return $win }
        }
    }
    return $null
}

# ---------------------------------------------------------------------------
# 0. Pre-flight
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "[0/5] Pre-flight checks"

$CondaExe = Find-Conda
if (-not $CondaExe) {
    Write-Error "conda not found. Install Miniconda/Anaconda and ensure its Scripts folder is in PATH."
    exit 1
}
Write-Host "  conda: $CondaExe"

if (-not (Test-Path $YmlFile)) { Write-Error "conda_env_setup.yml not found at $YmlFile"; exit 1 }
if (-not (Test-Path $ReqFile)) { Write-Error "requirements.txt not found at $ReqFile"; exit 1 }

# Locate vcvarsall.bat — prefer VS 2022 (v17) over newer versions
$VsWhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
$VcVarsAll = $null
if (Test-Path $VsWhere) {
    $vs2022 = & $VsWhere -products '*' -version '[17,18)' -property installationPath 2>$null | Select-Object -First 1
    if ($vs2022) {
        $candidate = Join-Path $vs2022 "VC\Auxiliary\Build\vcvarsall.bat"
        if (Test-Path $candidate) { $VcVarsAll = $candidate }
    }
    if (-not $VcVarsAll) {
        $anyVS = & $VsWhere -products '*' -property installationPath 2>$null | Select-Object -First 1
        if ($anyVS) {
            $candidate = Join-Path $anyVS "VC\Auxiliary\Build\vcvarsall.bat"
            if (Test-Path $candidate) { $VcVarsAll = $candidate }
        }
    }
}
if (-not $VcVarsAll) {
    Write-Host ""
    Write-Host "WARNING: Visual Studio Build Tools (C++ workload) not detected."
    Write-Host "  llama-cpp-python requires MSVC to build from source."
    Write-Host "  Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/"
    $ans = Read-Host "Continue anyway? [y/N]"
    if ($ans -notmatch '^[Yy]$') { exit 1 }
}

# ---------------------------------------------------------------------------
# 1. Create / update conda environment
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "[1/5] Creating / updating conda env '$EnvName' from $YmlFile"

$envList = & $CondaExe env list 2>$null
$envExists = $envList | Where-Object { $_ -match "^$EnvName\s" }
if ($envExists) {
    Write-Host "  Environment '$EnvName' exists – updating..."
    & $CondaExe env update --name $EnvName --file $YmlFile --prune
} else {
    & $CondaExe env create --name $EnvName --file $YmlFile
}

# ---------------------------------------------------------------------------
# Resolve python.exe
# ---------------------------------------------------------------------------
$PythonExe = Get-CondaEnvPython -CondaExe $CondaExe -Name $EnvName
if (-not $PythonExe) {
    Write-Error "Could not locate python.exe for env '$EnvName'."
    exit 1
}
Write-Host "  Using Python: $PythonExe"

# ---------------------------------------------------------------------------
# 2. Upgrade pip tooling
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "[2/5] Upgrading pip / setuptools / wheel"
& $PythonExe -m pip install --upgrade pip wheel "setuptools<82"

# ---------------------------------------------------------------------------
# 3. Detect CUDA and install PyTorch
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "[3/5] Installing PyTorch"

function Get-CudaVersion {
    # Prefer nvcc (toolkit version) over nvidia-smi (driver max version)
    $nvccPath = $null
    $cmd = Get-Command nvcc -ErrorAction SilentlyContinue
    if ($cmd) {
        $nvccPath = $cmd.Source
    } else {
        $found = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -Filter "nvcc.exe" -Recurse -ErrorAction SilentlyContinue |
                 Select-Object -First 1
        if ($found) { $nvccPath = $found.FullName }
    }
    if ($nvccPath) {
        $output = & $nvccPath --version 2>$null
        $match  = $output | Select-String 'release\s+([0-9]+\.[0-9]+)'
        if ($match) { return $match.Matches[0].Groups[1].Value.Trim() }
    }
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smi) {
        $line = & nvidia-smi 2>$null | Select-String 'CUDA Version'
        if ($line) {
            return ($line -replace '.*CUDA Version: ([0-9]+\.[0-9]+).*', '$1').Trim()
        }
    }
    return $null
}

function Get-TorchTag([string]$CudaVer) {
    if (-not $CudaVer) { return "cpu" }
    $parts = $CudaVer -split '\.'
    $major = [int]$parts[0]; $minor = [int]$parts[1]
    if     ($major -ge 13)                    { return "cu128" }
    elseif ($major -eq 12 -and $minor -ge 8)  { return "cu128" }
    elseif ($major -eq 12 -and $minor -ge 6)  { return "cu126" }
    elseif ($major -eq 12 -and $minor -ge 4)  { return "cu124" }
    elseif ($major -eq 12 -and $minor -ge 1)  { return "cu121" }
    elseif ($major -eq 11 -and $minor -ge 8)  { return "cu118" }
    else                                       { return "cpu"   }
}

$CudaVersion = Get-CudaVersion
$TorchTag = Get-TorchTag -CudaVer $CudaVersion

if ($CudaVersion) {
    Write-Host "  Detected CUDA: $CudaVersion  ->  PyTorch tag: $TorchTag"
} else {
    Write-Host "  No CUDA detected – installing CPU-only PyTorch."
}

if ($TorchTag -eq "cpu") {
    & $PythonExe -m pip install --upgrade torch torchvision torchaudio
} else {
    & $PythonExe -m pip install --upgrade torch torchvision torchaudio `
        --index-url "https://download.pytorch.org/whl/$TorchTag"
}

# ---------------------------------------------------------------------------
# 4. Install llama-cpp-python
# ---------------------------------------------------------------------------
$LlamaVerSpec = if ($LlamaCppVersion) { "==$LlamaCppVersion" } else { "" }
Write-Host ""
Write-Host "[4/5] Installing llama-cpp-python$LlamaVerSpec"

try { & $PythonExe -m pip uninstall -y llama-cpp-python 2>&1 | Out-Null } catch { <# not installed #> }

if ($TorchTag -eq "cpu") {
    & $PythonExe -m pip install --upgrade "llama-cpp-python$LlamaVerSpec"
} else {
    # Try prebuilt wheel index first (Windows wheels only exist up to 0.3.4)
    $CudaShort = "cu" + ($CudaVersion -replace '\.', '')
    $WheelIndex = "https://abetlen.github.io/llama-cpp-python/whl/$CudaShort"
    Write-Host "  Trying pre-built wheel from: $WheelIndex"

    $wheelInstalled = $false
    try {
        & $PythonExe -m pip install --upgrade `
            --only-binary llama-cpp-python `
            --extra-index-url $WheelIndex `
            "llama-cpp-python$LlamaVerSpec"
        $wheelInstalled = ($LASTEXITCODE -eq 0)
    } catch { $wheelInstalled = $false }

    if ($wheelInstalled) {
        Write-Host "  Pre-built wheel installed."
    } else {
        Write-Host "  No pre-built wheel found – source build required."

        if (-not $VcVarsAll) {
            Write-Error "vcvarsall.bat not found. Install VS Build Tools with 'Desktop development with C++'."
            exit 1
        }

        # Install build tools into the conda env
        & $PythonExe -m pip install --upgrade "cmake<4" ninja scikit-build-core

        $EnvScripts = Join-Path (Split-Path $PythonExe) "Scripts"
        $CmakeExe   = Join-Path $EnvScripts "cmake.exe"
        $NinjaExe   = Join-Path $EnvScripts "ninja.exe"
        if (-not (Test-Path $CmakeExe)) { $CmakeExe = (Get-Command cmake -ErrorAction Stop).Source }
        if (-not (Test-Path $NinjaExe)) { $NinjaExe = (Get-Command ninja -ErrorAction Stop).Source }

        Write-Host "  Importing MSVC environment from: $VcVarsAll"
        Import-VcVarsAll -VcVarsAllPath $VcVarsAll

        # /utf-8: fixes builds on non-UTF-8 system locales (e.g. code page 950)
        $env:CL                       = "/utf-8"
        $env:SKBUILD_CMAKE_EXECUTABLE  = $CmakeExe
        $env:SKBUILD_CMAKE_GENERATOR   = "Ninja"
        $env:CMAKE_GENERATOR           = "Ninja"
        $env:CMAKE_ARGS                = "-DGGML_CUDA=on -DCMAKE_MAKE_PROGRAM=`"$NinjaExe`" -DCMAKE_CXX_FLAGS=/utf-8 -DCMAKE_C_FLAGS=/utf-8"
        $env:FORCE_CMAKE               = "1"

        Write-Host "  Building llama-cpp-python$LlamaVerSpec (CUDA + Ninja)..."
        & $PythonExe -m pip install --no-cache-dir `
            --no-binary llama-cpp-python `
            --no-build-isolation `
            "llama-cpp-python$LlamaVerSpec"

        if ($LASTEXITCODE -ne 0) {
            Write-Error "llama-cpp-python source build failed."
            exit 1
        }
    }
}

# ---------------------------------------------------------------------------
# 5. Install remaining requirements (skip torch / llama-cpp-python lines)
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "[5/5] Installing requirements from $ReqFile"

$FilteredReq = [System.IO.Path]::GetTempFileName() + ".txt"
Get-Content $ReqFile |
    Where-Object { $_ -notmatch '^\s*(llama-cpp-python|torch|torchvision|torchaudio)\b' } |
    Set-Content $FilteredReq

& $PythonExe -m pip install -r $FilteredReq
Remove-Item $FilteredReq -ErrorAction SilentlyContinue

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "Verifying installs..."
& $PythonExe -c @"
import torch, llama_cpp
print('torch.cuda.is_available =', torch.cuda.is_available())
print('torch.version.cuda      =', getattr(torch.version, 'cuda', None))
print('llama_cpp version       =', getattr(llama_cpp, '__version__', 'unknown'))
fn = getattr(llama_cpp.llama_cpp, 'llama_supports_gpu_offload', None)
print('llama_supports_gpu_offload =', bool(fn()) if callable(fn) else '<not exposed>')
"@

Write-Host ""
Write-Host "Setup complete."
