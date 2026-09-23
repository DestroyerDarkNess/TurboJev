[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$AllArchitectures = @("x86", "x64", "arm64")
$RustTargets = @{
    x86 = "i686-pc-windows-msvc"
    x64 = "x86_64-pc-windows-msvc"
    arm64 = "aarch64-pc-windows-msvc"
}

function Import-DotEnv {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    foreach ($line in Get-Content -LiteralPath $Path) {
        $trimmed = $line.Trim()
        if ([string]::IsNullOrWhiteSpace($trimmed) -or $trimmed.StartsWith("#")) {
            continue
        }

        $pair = $trimmed -split "=", 2
        if ($pair.Count -ne 2) {
            continue
        }

        $name = $pair[0].Trim()
        $value = $pair[1].Trim()
        if ($value.Length -ge 2 -and (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'")))) {
            $value = $value.Substring(1, $value.Length - 2)
        }

        [Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
}

function Invoke-Tool {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [string]$WorkingDirectory = $Root,
        [string[]]$DisplayArguments = $Arguments
    )

    Push-Location $WorkingDirectory
    try {
        Write-Host ("> {0} {1}" -f $Command, ($DisplayArguments -join " ")) -ForegroundColor DarkGray
        & $Command @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code ${LASTEXITCODE}: $Command"
        }
    }
    finally {
        Pop-Location
    }
}

function Require-Value {
    param([string]$Name)

    $value = [Environment]::GetEnvironmentVariable($Name, "Process")
    if ([string]::IsNullOrWhiteSpace($value)) {
        throw "$Name is missing from .env"
    }
    return $value
}

function Publish-Npm {
    Require-Value "NPM_TOKEN" | Out-Null
    $npm = if (Get-Command npm.cmd -ErrorAction SilentlyContinue) { "npm.cmd" } else { "npm" }
    $tempNpmrc = Join-Path $env:TEMP ("turbojev-npm-{0}.npmrc" -f [guid]::NewGuid().ToString("N"))
    $previousConfig = $env:NPM_CONFIG_USERCONFIG

    try {
        Set-Content -LiteralPath $tempNpmrc -Value ("//registry.npmjs.org/:_authToken={0}" -f $env:NPM_TOKEN) -NoNewline
        $env:NPM_CONFIG_USERCONFIG = $tempNpmrc

        foreach ($package in @("bindings/web", "bindings/web-runtimes/transformersjs")) {
            Invoke-Tool $npm @("install", "--ignore-scripts", "--no-audit", "--no-fund") (Join-Path $Root $package)
            Invoke-Tool $npm @("run", "build") (Join-Path $Root $package)
            Invoke-Tool $npm @("publish", "--access", "public") (Join-Path $Root $package)
        }
    }
    finally {
        Remove-Item -LiteralPath $tempNpmrc -Force -ErrorAction SilentlyContinue
        if ($null -eq $previousConfig) {
            Remove-Item Env:NPM_CONFIG_USERCONFIG -ErrorAction SilentlyContinue
        }
        else {
            $env:NPM_CONFIG_USERCONFIG = $previousConfig
        }
    }
}

function Publish-Python {
    Require-Value "PYPI_API_TOKEN" | Out-Null
    $output = Join-Path $Root "artifacts/pypi"
    New-Item -ItemType Directory -Force -Path $output | Out-Null
    Get-ChildItem -LiteralPath $output -Filter "*.whl" -ErrorAction SilentlyContinue | Remove-Item -Force
    $env:TWINE_USERNAME = "__token__"
    $env:TWINE_PASSWORD = $env:PYPI_API_TOKEN

    try {
        Invoke-Tool "python" @("-m", "pip", "install", "build", "twine")
        foreach ($architecture in $AllArchitectures) {
            $target = $RustTargets[$architecture]
            Invoke-Tool "rustup" @("target", "add", $target)
            Invoke-Tool "python" @("-m", "maturin", "build", "--release", "--target", $target, "--out", $output) (Join-Path $Root "bindings/python")
        }

        $packages = Get-ChildItem -LiteralPath $output -Filter "*.whl"
        if (-not $packages) {
            throw "No Python wheel was produced"
        }
        Invoke-Tool "python" (@("-m", "twine", "upload") + @($packages.FullName))
    }
    finally {
        Remove-Item Env:TWINE_USERNAME -ErrorAction SilentlyContinue
        Remove-Item Env:TWINE_PASSWORD -ErrorAction SilentlyContinue
    }
}

function Publish-NuGet {
    Require-Value "NUGET_API_KEY" | Out-Null
    $nativeRoot = Join-Path $Root "artifacts/nuget-native"
    $output = Join-Path $Root "artifacts/nuget"
    New-Item -ItemType Directory -Force -Path $nativeRoot | Out-Null
    New-Item -ItemType Directory -Force -Path $output | Out-Null

    foreach ($architecture in $AllArchitectures) {
        $target = $RustTargets[$architecture]
        $rid = "win-$architecture"
        $ridRoot = Join-Path $nativeRoot $rid
        New-Item -ItemType Directory -Force -Path $ridRoot | Out-Null
        Invoke-Tool "rustup" @("target", "add", $target)
        Invoke-Tool "cargo" @("build", "--locked", "--release", "--target", $target, "-p", "turbojev-capi")
        $native = Join-Path $Root "target/$target/release/turbojev_capi.dll"
        if (-not (Test-Path -LiteralPath $native)) {
            throw "Native C API was not built for $rid"
        }
        Copy-Item -LiteralPath $native -Destination (Join-Path $ridRoot "turbojev_capi.dll") -Force
    }

    Invoke-Tool "dotnet" @("pack", "bindings/dotnet/TurboJev/TurboJev.csproj", "--configuration", "Release", "-p:TurboJevNativeRoot=$nativeRoot", "--output", $output)

    foreach ($package in Get-ChildItem -LiteralPath $output -Filter "*.nupkg") {
        $arguments = @("nuget", "push", $package.FullName, "--api-key", $env:NUGET_API_KEY, "--source", "https://api.nuget.org/v3/index.json", "--skip-duplicate")
        $display = @("nuget", "push", $package.FullName, "--api-key", "***", "--source", "https://api.nuget.org/v3/index.json", "--skip-duplicate")
        Invoke-Tool -Command "dotnet" -Arguments $arguments -DisplayArguments $display
    }
}

function Publish-Cargo {
    Require-Value "CARGO_REGISTRY_TOKEN" | Out-Null
    $previousToken = $env:CARGO_REGISTRIES_CRATES_IO_TOKEN
    $env:CARGO_REGISTRIES_CRATES_IO_TOKEN = $env:CARGO_REGISTRY_TOKEN

    try {
        Write-Host "Cargo preflight (the workspace must have publishable versioned dependencies)..." -ForegroundColor Yellow
        Invoke-Tool "cargo" @("publish", "--dry-run", "--locked", "-p", "turbojev")
        Invoke-Tool "cargo" @("publish", "--locked", "-p", "turbojev")
    }
    finally {
        if ($null -eq $previousToken) {
            Remove-Item Env:CARGO_REGISTRIES_CRATES_IO_TOKEN -ErrorAction SilentlyContinue
        }
        else {
            $env:CARGO_REGISTRIES_CRATES_IO_TOKEN = $previousToken
        }
    }
}

function Confirm-Publish {
    $answer = Read-Host "This uploads packages to public registries. Type PUBLISH to continue"
    return $answer -ceq "PUBLISH"
}

Import-DotEnv (Join-Path $Root ".env")

if (-not (Test-Path -LiteralPath (Join-Path $Root ".env"))) {
    Copy-Item -LiteralPath (Join-Path $Root ".env.example") -Destination (Join-Path $Root ".env")
    Write-Warning "Created .env from .env.example. Add credentials and run the script again."
    exit 1
}

Write-Host "TurboJev local publisher" -ForegroundColor Cyan
Write-Host "Credentials are read from .env and are never committed."

while ($true) {
    Write-Host ""
    Write-Host "Publish target" -ForegroundColor Cyan
    Write-Host "  1) npm"
    Write-Host "  2) PyPI"
    Write-Host "  3) NuGet"
    Write-Host "  4) crates.io"
    Write-Host "  5) all"
    Write-Host "  6) exit"

    $choice = Read-Host "Select target"
    if ($choice -eq "6") {
        break
    }

    try {
        if (-not (Confirm-Publish)) {
            Write-Host "Cancelled."
            continue
        }

        switch ($choice) {
            "1" {
                Write-Host "npm web packages are architecture-independent; publishing once."
                Publish-Npm
            }
            "2" { Publish-Python }
            "3" {
                Write-Host "Building and publishing one NuGet package with x86, x64, and arm64 runtime assets."
                Publish-NuGet
            }
            "4" {
                Write-Host "Cargo crates are source packages; publishing once."
                Publish-Cargo
            }
            "5" {
                $failures = [System.Collections.Generic.List[string]]::new()
                foreach ($target in @("npm", "python", "nuget", "cargo")) {
                    try {
                        switch ($target) {
                            "npm" { Publish-Npm }
                            "python" { Publish-Python }
                            "nuget" { Publish-NuGet }
                            "cargo" { Publish-Cargo }
                        }
                    }
                    catch {
                        $failures.Add($target)
                        Write-Warning ("{0} failed: {1}" -f $target, $_.Exception.Message)
                    }
                }
                if ($failures.Count -gt 0) {
                    throw ("Targets failed: " + ($failures -join ", "))
                }
            }
            default { throw "Invalid target selection" }
        }

        Write-Host "Completed successfully." -ForegroundColor Green
    }
    catch {
        Write-Error $_
    }
}
