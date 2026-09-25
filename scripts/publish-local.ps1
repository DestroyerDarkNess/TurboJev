[CmdletBinding()]
param(
    [ValidateSet('npm', 'python', 'nuget', 'cargo', 'all')][string]$Target,
    [switch]$Yes
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
. (Join-Path $PSScriptRoot 'use-arm64-toolchain.ps1')
$AllArchitectures = @("x86", "x64", "arm64")
$RustTargets = @{
    x86 = "i686-pc-windows-msvc"
    x64 = "x86_64-pc-windows-msvc"
    arm64 = "aarch64-pc-windows-msvc"
}
$NodeArchitectures = @{
    x86 = "ia32"
    x64 = "x64"
    arm64 = "arm64"
}
$LlamaPlan = Get-Content (Join-Path $Root "tests/e2e/final-plan.json") -Raw | ConvertFrom-Json
$LlamaSource = Join-Path $Root ".cache/real-validation/llama.cpp"
$NativeRuntimeRoot = Join-Path $Root "artifacts/native-runtime"
$ReleaseVersion = [regex]::Match((Get-Content (Join-Path $Root "Cargo.toml") -Raw), '(?m)^version = "([0-9.]+)"').Groups[1].Value

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

function Ensure-LlamaCppSource {
    if (-not (Test-Path -LiteralPath (Join-Path $LlamaSource "CMakeLists.txt"))) {
        New-Item -ItemType Directory -Force -Path (Split-Path $LlamaSource) | Out-Null
        Invoke-Tool "git" @("clone", "--depth", "1", "--branch", $LlamaPlan.llamaCpp.tag, $LlamaPlan.llamaCpp.repository, $LlamaSource)
    }
    $revision = (& git -C $LlamaSource rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $revision.StartsWith($LlamaPlan.llamaCpp.commitPrefix)) {
        throw "Cached llama.cpp checkout does not match pinned revision $($LlamaPlan.llamaCpp.commitPrefix)."
    }
    $script:LlamaCppRevision = $revision
}

function Build-NativeRuntime {
    param([Parameter(Mandatory = $true)][string]$Architecture)

    $rid = "win-$Architecture"
    $runtimeDirectory = Join-Path $NativeRuntimeRoot $rid
    $shimLib = Join-Path $runtimeDirectory "turbojev_llama_shim.lib"
    $requiredDlls = @("turbojev_llama_shim.dll", "llama.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll")
    $enableVulkan = $Architecture -in @("x64", "arm64")
    $vulkanLibrary = $null
    if ($enableVulkan) {
        if ([string]::IsNullOrWhiteSpace($env:VULKAN_SDK)) { throw "VULKAN_SDK is required to build the GPU runtime for $rid." }
        $vulkanLibDir = if ($Architecture -eq "arm64") { Join-Path $env:VULKAN_SDK "Lib-ARM64" } else { Join-Path $env:VULKAN_SDK "Lib" }
        $vulkanLibrary = Join-Path $vulkanLibDir "vulkan-1.lib"
        if (-not (Test-Path -LiteralPath (Join-Path $env:VULKAN_SDK "Include/vulkan/vulkan.h")) -or -not (Test-Path -LiteralPath $vulkanLibrary)) {
            throw "The Vulkan SDK does not contain headers and a $Architecture import library for $rid."
        }
        $requiredDlls += "ggml-vulkan.dll"
    }
    $markerPath = Join-Path $runtimeDirectory "runtime-build.json"
    $markerMatches = $false
    if (Test-Path -LiteralPath $markerPath) {
        $marker = Get-Content $markerPath -Raw | ConvertFrom-Json
        $markerMatches = $marker.turbojevVersion -eq $ReleaseVersion -and $marker.llamaCppRevision.StartsWith($LlamaPlan.llamaCpp.commitPrefix) -and $marker.vulkanEnabled -eq $enableVulkan
    }
    $missingRuntimeDlls = @($requiredDlls | Where-Object { -not (Test-Path -LiteralPath (Join-Path $runtimeDirectory $_)) })
    if ($markerMatches -and (Test-Path -LiteralPath $shimLib) -and $missingRuntimeDlls.Count -eq 0) {
        $env:TURBOJEV_LLAMA_SHIM_LIB_DIR = $runtimeDirectory
        return
    }

    Ensure-LlamaCppSource
    $revision = $script:LlamaCppRevision
    $buildDirectory = Join-Path $Root "artifacts/llama-build/$rid"
    New-Item -ItemType Directory -Force -Path $runtimeDirectory | Out-Null
    $cmakeArguments = @(
        "--fresh",
        "-S", (Join-Path $Root "crates/turbojev-runtime-llamacpp/native"),
        "-B", $buildDirectory,
        "-DLLAMA_CPP_SOURCE_DIR=$LlamaSource",
        "-DGGML_VULKAN=$(if ($enableVulkan) { 'ON' } else { 'OFF' })",
        "-DGGML_OPENMP=OFF",
        "-DLLAMA_OPENSSL=OFF"
    )
    if ($enableVulkan) {
        $cmakeArguments += @(
            "-DVulkan_INCLUDE_DIR=$(Join-Path $env:VULKAN_SDK 'Include')",
            "-DVulkan_LIBRARY=$vulkanLibrary"
        )
    }
    $restoreEnvironment = $null
    if ($Architecture -eq "arm64") {
        $restoreEnvironment = @{}
        Get-ChildItem Env: | ForEach-Object { $restoreEnvironment[$_.Name] = $_.Value }
        $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio/Installer/vswhere.exe"
        $vsInstall = (& $vswhere -latest -products "*" -requires Microsoft.VisualStudio.Component.VC.Tools.ARM64 -property installationPath).Trim()
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($vsInstall)) { throw "Visual Studio ARM64 C++ tools were not found." }
        $vcvars = Join-Path $vsInstall "VC/Auxiliary/Build/vcvarsall.bat"
        $clang = Join-Path $vsInstall "VC/Tools/Llvm/bin/clang-cl.exe"
        if (-not (Test-Path -LiteralPath $clang)) { throw "Visual Studio clang-cl was not found: $clang" }
        $environmentLines = & $env:ComSpec /d /s /c ('"{0}" x64_arm64 > nul && set' -f $vcvars)
        if ($LASTEXITCODE -ne 0) { throw "vcvarsall failed to configure the ARM64 compiler environment." }
        foreach ($line in $environmentLines) {
            if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process") }
        }
        $buildDirectory = Join-Path $Root "artifacts/llama-build/$rid-clang"
        $cmakeArguments += @(
            "-G", "Ninja",
            "-DCMAKE_C_COMPILER=$clang",
            "-DCMAKE_CXX_COMPILER=$clang",
            "-DCMAKE_C_COMPILER_TARGET=arm64-pc-windows-msvc",
            "-DCMAKE_CXX_COMPILER_TARGET=arm64-pc-windows-msvc",
            "-DCMAKE_SYSTEM_NAME=Windows",
            "-DCMAKE_SYSTEM_PROCESSOR=ARM64",
            "-DCMAKE_BUILD_TYPE=Release"
        )
        $cmakeArguments[3] = "-B"
        $cmakeArguments[4] = $buildDirectory
    } else {
        $platform = @{ x86 = "Win32"; x64 = "x64" }[$Architecture]
        $cmakeArguments += @("-A", $platform)
    }
    try {
        Invoke-Tool "cmake" $cmakeArguments
        Invoke-Tool "cmake" @("--build", $buildDirectory, "--config", "Release", "--target", "turbojev_llama_shim", "--parallel", "4")
    } finally {
        if ($null -ne $restoreEnvironment) {
            foreach ($item in Get-ChildItem Env:) {
                if (-not $restoreEnvironment.ContainsKey($item.Name)) { Remove-Item "Env:$($item.Name)" -ErrorAction SilentlyContinue }
            }
            foreach ($name in $restoreEnvironment.Keys) { [Environment]::SetEnvironmentVariable($name, $restoreEnvironment[$name], "Process") }
        }
    }

    foreach ($library in Get-ChildItem -LiteralPath $buildDirectory -Filter "*.dll" -File -Recurse) {
        Copy-Item -LiteralPath $library.FullName -Destination (Join-Path $runtimeDirectory $library.Name) -Force
    }
    $shimImportLibrary = Get-ChildItem -LiteralPath $buildDirectory -Filter "turbojev_llama_shim.lib" -File -Recurse | Select-Object -First 1
    if (-not $shimImportLibrary) { throw "CMake did not produce the llama shim import library for $rid." }
    Copy-Item -LiteralPath $shimImportLibrary.FullName -Destination $shimLib -Force
    $missingDlls = @($requiredDlls | Where-Object { -not (Test-Path -LiteralPath (Join-Path $runtimeDirectory $_)) })
    if ($missingDlls.Count -gt 0) { throw "CMake runtime for $rid is incomplete: $($missingDlls -join ', ')." }
    @{ turbojevVersion = $ReleaseVersion; llamaCppRevision = $revision; target = $rid; vulkanEnabled = $enableVulkan } |
        ConvertTo-Json | Set-Content -LiteralPath $markerPath -Encoding utf8

    $env:TURBOJEV_LLAMA_SHIM_LIB_DIR = $runtimeDirectory
}

function Publish-NpmPackage {
    param([string]$NpmCommand, [string]$PackagePath)
    $manifest = Get-Content -Raw (Join-Path $PackagePath 'package.json') | ConvertFrom-Json
    $uri = 'https://registry.npmjs.org/' + [uri]::EscapeDataString($manifest.name) + '/' + $manifest.version
    try {
        $published = Invoke-RestMethod $uri
        if ($published.version -eq $manifest.version) {
            Write-Host "Already published: $($manifest.name)@$($manifest.version)"
            return
        }
    }
    catch {
        if ([int]$_.Exception.Response.StatusCode -ne 404) { throw }
    }
    Invoke-Tool $NpmCommand @('publish', '--access', 'public') $PackagePath
}

function Publish-Node {
    param([string]$NpmCommand)

    $nodePackage = Join-Path $Root "bindings/node"
    $output = Join-Path $Root "artifacts/npm-native"
    $manifest = Get-Content -Raw (Join-Path $nodePackage "package.json") | ConvertFrom-Json
    $platformPackages = @()
    if (Test-Path -LiteralPath $output) { Remove-Item -LiteralPath $output -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $output | Out-Null

    Invoke-Tool $NpmCommand @("install", "--ignore-scripts", "--no-audit", "--no-fund", "--package-lock=false", "--omit=optional") $nodePackage
    Invoke-Tool $NpmCommand @("run", "build") $nodePackage

    foreach ($architecture in $AllArchitectures) {
        $target = $RustTargets[$architecture]
        $nodeArchitecture = $NodeArchitectures[$architecture]
        Build-NativeRuntime $architecture
        $runtimeDirectory = Join-Path $NativeRuntimeRoot "win-$architecture"
        Invoke-Tool "rustup" @("target", "add", $target)
        Invoke-Tool "cargo" @("build", "--locked", "--release", "--target", $target, "-p", "turbojev-node", "--features", "turbojev-node/llamacpp")

        $native = Join-Path $Root "target/$target/release/turbojev_node.dll"
        if (-not (Test-Path -LiteralPath $native)) {
            throw "Node native addon was not built for win32-$nodeArchitecture"
        }

        $platformPackage = Join-Path $output ("node-win32-{0}" -f $nodeArchitecture)
        New-Item -ItemType Directory -Force -Path $platformPackage | Out-Null
        Copy-Item -LiteralPath $native -Destination (Join-Path $platformPackage "turbojev.node") -Force
        Get-ChildItem -LiteralPath $runtimeDirectory -Filter "*.dll" -File | Copy-Item -Destination $platformPackage -Force
        $platformManifest = [ordered]@{
            name = "@turbojev/node-win32-$nodeArchitecture"
            version = $manifest.version
            description = "TurboJev native Node.js addon for win32-$nodeArchitecture"
            main = "turbojev.node"
            os = @("win32")
            cpu = @($nodeArchitecture)
            license = $manifest.license
            repository = $manifest.repository
        }
        $platformManifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $platformPackage "package.json")
        Invoke-Tool "python" @("scripts/check-native-node-package.py", $platformPackage)
        $platformPackages += $platformPackage
    }

    $realModel = Join-Path $Root ".cache/real-validation/$($LlamaPlan.model.filename)"
    if (-not (Test-Path -LiteralPath $realModel)) {
        Invoke-Tool "python" @("scripts/download_model.py", "--plan", "tests/e2e/final-plan.json")
    }
    $hostAddon = Join-Path $output "node-win32-x64/turbojev.node"
    Invoke-Tool "node" @("tests/e2e/node/real_smoke.mjs", $realModel, $hostAddon)
    Invoke-Tool $NpmCommand @("test") $nodePackage

    foreach ($platformPackage in $platformPackages) {
        Publish-NpmPackage $NpmCommand $platformPackage
    }
    Publish-NpmPackage $NpmCommand $nodePackage
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
            Invoke-Tool $npm @("install", "--ignore-scripts", "--no-audit", "--no-fund", "--package-lock=false") (Join-Path $Root $package)
            Invoke-Tool $npm @("run", "build") (Join-Path $Root $package)
            Publish-NpmPackage $npm (Join-Path $Root $package)
        }
        Publish-Node $npm
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
    if (Test-Path -LiteralPath $output) {
        Get-ChildItem -LiteralPath $output -Directory | Remove-Item -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $output | Out-Null
    Get-ChildItem -LiteralPath $output -Filter "*.whl" -File -Recurse -ErrorAction SilentlyContinue | Remove-Item -Force
    $env:TWINE_USERNAME = "__token__"
    $env:TWINE_PASSWORD = $env:PYPI_API_TOKEN
    $previousPyConfig = $env:PYO3_CONFIG_FILE

    try {
        $toolsRoot = Join-Path $Root "artifacts/publish-tools"
        $python = Join-Path $toolsRoot "Scripts/python.exe"
        if (-not (Test-Path $python)) {
            Invoke-Tool "python" @("-m", "venv", $toolsRoot)
        }
        Invoke-Tool $python @("-m", "pip", "install", "maturin>=1.7,<2", "twine")
        foreach ($architecture in $AllArchitectures) {
            $target = $RustTargets[$architecture]
            Build-NativeRuntime $architecture
            $runtimeDirectory = Join-Path $NativeRuntimeRoot "win-$architecture"
            Invoke-Tool "rustup" @("target", "add", $target)
            $config = Join-Path $output "pyo3-$architecture.txt"
            $width = if ($architecture -eq 'x86') { 32 } else { 64 }
            @("implementation=CPython", "version=3.9", "shared=true", "abi3=true", "lib_name=python3", "pointer_width=$width", "ext_suffix=.pyd") | Set-Content $config
            if ($architecture -eq 'arm64' -and $env:TURBOJEV_ARM64_LIB_EXE) {
                $metadata = (& cargo metadata --format-version 1 --locked) | ConvertFrom-Json
                if ($LASTEXITCODE -ne 0) { throw 'Could not read Cargo metadata' }
                $importPackage = $metadata.packages | Where-Object name -EQ 'python3-dll-a'
                $definition = Join-Path (Split-Path $importPackage.manifest_path) 'src/python3.def'
                $importDir = Join-Path $output 'arm64-import'
                New-Item -ItemType Directory -Force $importDir | Out-Null
                Invoke-Tool $env:TURBOJEV_ARM64_LIB_EXE @('/NOLOGO', '/MACHINE:ARM64', "/DEF:$definition", "/OUT:$importDir/python3.lib")
                "lib_dir=$importDir" | Add-Content $config
            }
            $env:PYO3_CONFIG_FILE = $config
            $architectureOutput = Join-Path $output $architecture
            New-Item -ItemType Directory -Force -Path $architectureOutput | Out-Null
            Invoke-Tool $python @("-m", "maturin", "build", "--locked", "--release", "--target", $target, "--features", "llamacpp", "--out", $architectureOutput) (Join-Path $Root "bindings/python")
            $wheel = Get-ChildItem -LiteralPath $architectureOutput -Filter "*.whl" | Select-Object -First 1
            if (-not $wheel) { throw "Maturin did not produce a wheel for $architecture." }
            Invoke-Tool $python @("scripts/bundle-native-wheel.py", $wheel.FullName, $runtimeDirectory)
        }

        $packages = Get-ChildItem -LiteralPath $output -Filter "*.whl" -File -Recurse
        if (-not $packages) {
            throw "No Python wheel was produced"
        }
        if ($packages.Count -ne $AllArchitectures.Count) { throw "Expected one wheel per architecture" }
        Invoke-Tool $python (@("-m", "twine", "check") + @($packages.FullName))
        $hostWheel = $packages | Where-Object Name -Like '*win_amd64.whl'
        Invoke-Tool $python @('-m', 'pip', 'install', '--force-reinstall', '--no-deps', $hostWheel.FullName)
        Invoke-Tool $python @('scripts/smoke-python-package.py')
        $realModel = Join-Path $Root ".cache/real-validation/$($LlamaPlan.model.filename)"
        if (-not (Test-Path -LiteralPath $realModel)) {
            Invoke-Tool $python @("scripts/download_model.py", "--plan", "tests/e2e/final-plan.json")
        }
        Invoke-Tool $python @("tests/e2e/python/real_smoke.py", $realModel, "--report", (Join-Path $output "real-smoke.json"))
        Invoke-Tool $python (@("-m", "twine", "upload", "--non-interactive", "--skip-existing") + @($packages.FullName))
    }
    finally {
        $env:PYO3_CONFIG_FILE = $previousPyConfig
        Remove-Item Env:TWINE_USERNAME -ErrorAction SilentlyContinue
        Remove-Item Env:TWINE_PASSWORD -ErrorAction SilentlyContinue
    }
}

function Publish-NuGet {
    Require-Value "NUGET_API_KEY" | Out-Null
    $nativeRoot = Join-Path $Root "artifacts/nuget-native"
    $output = Join-Path $Root "artifacts/nuget"
    if (Test-Path -LiteralPath $nativeRoot) { Remove-Item -LiteralPath $nativeRoot -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $nativeRoot | Out-Null
    New-Item -ItemType Directory -Force -Path $output | Out-Null
    Get-ChildItem -LiteralPath $nativeRoot -Directory -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
    Get-ChildItem -LiteralPath $output -Filter "*.nupkg" -File -ErrorAction SilentlyContinue | Remove-Item -Force

    foreach ($architecture in $AllArchitectures) {
        $target = $RustTargets[$architecture]
        $rid = "win-$architecture"
        $ridRoot = Join-Path $nativeRoot $rid
        New-Item -ItemType Directory -Force -Path $ridRoot | Out-Null
        Build-NativeRuntime $architecture
        $runtimeDirectory = Join-Path $NativeRuntimeRoot $rid
        Invoke-Tool "rustup" @("target", "add", $target)
        Invoke-Tool "cargo" @("build", "--locked", "--release", "--target", $target, "-p", "turbojev-capi", "--features", "turbojev-capi/llamacpp")
        $native = Join-Path $Root "target/$target/release/turbojev_capi.dll"
        if (-not (Test-Path -LiteralPath $native)) {
            throw "Native C API was not built for $rid"
        }
        Copy-Item -LiteralPath $native -Destination (Join-Path $ridRoot "turbojev_capi.dll") -Force
        Get-ChildItem -LiteralPath $runtimeDirectory -Filter "*.dll" -File | Copy-Item -Destination $ridRoot -Force
    }

    Invoke-Tool "dotnet" @("pack", "bindings/dotnet/TurboJev/TurboJev.csproj", "--configuration", "Release", "-p:TurboJevNativeRoot=$nativeRoot", "--output", $output)

    $package = Get-ChildItem -LiteralPath $output -Filter "TurboJev.*.nupkg" -File | Select-Object -First 1
    if (-not $package) { throw "NuGet package was not created." }
    Invoke-Tool "python" @("scripts/check-native-nuget.py", $package.FullName)
    $realModel = Join-Path $Root ".cache/real-validation/$($LlamaPlan.model.filename)"
    if (-not (Test-Path -LiteralPath $realModel)) {
        Invoke-Tool "python" @("scripts/download_model.py", "--plan", "tests/e2e/final-plan.json")
    }
    Invoke-Tool "pwsh" @("-NoProfile", "-File", "scripts/smoke-nuget-package.ps1", $package.FullName, $realModel)

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
        Invoke-Tool "python" @("scripts/publish-crates.py")
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
    if ($Yes) { return $true }
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

    $choice = if ($Target) { @{npm='1';python='2';nuget='3';cargo='4';all='5'}[$Target] } else { Read-Host "Select target" }
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
        if ($Target) { break }
    }
    catch {
        Write-Error $_
    }
}
