[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$PackagePath,
    [string]$ModelPath
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$PackagePath = (Resolve-Path -LiteralPath $PackagePath).Path
$Version = [regex]::Match([IO.Path]::GetFileName($PackagePath), '^TurboJev\.(.+)\.nupkg$').Groups[1].Value
if (!$Version) { throw 'Cannot determine the NuGet package version.' }
if ($ModelPath) { $ModelPath = (Resolve-Path -LiteralPath $ModelPath).Path }

# Isolate caches by package content so retries never test a stale registry build.
$hash = (Get-FileHash -LiteralPath $PackagePath -Algorithm SHA256).Hash.Substring(0, 12)
$SmokeRoot = Join-Path $Root "artifacts/dotnet-compat/$hash"
New-Item -ItemType Directory -Force -Path $SmokeRoot | Out-Null
$Project = Join-Path $Root 'tests/e2e/dotnet-compat/DotnetCompat.csproj'
$Config = Join-Path $SmokeRoot 'NuGet.Config'
$source = [Security.SecurityElement]::Escape((Split-Path -Parent $PackagePath))
@"
<configuration>
  <packageSources>
    <clear />
    <add key="local" value="$source" />
    <add key="nuget.org" value="https://api.nuget.org/v3/index.json" />
  </packageSources>
  <packageSourceMapping>
    <packageSource key="local"><package pattern="TurboJev" /></packageSource>
    <packageSource key="nuget.org"><package pattern="*" /></packageSource>
  </packageSourceMapping>
</configuration>
"@ | Set-Content -LiteralPath $Config

function Invoke-Checked {
    param([string]$Command, [string[]]$Arguments)
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Command failed with exit code $LASTEXITCODE" }
}

$properties = @(
    "-p:TurboJevPackageVersion=$Version",
    "-p:RestorePackagesPath=$SmokeRoot/packages",
    "-p:BaseIntermediateOutputPath=$SmokeRoot/obj/",
    "-p:BaseOutputPath=$SmokeRoot/bin/"
)
Invoke-Checked dotnet (@('restore', $Project, '--configfile', $Config, '-v', 'minimal') + $properties)
Invoke-Checked dotnet (@('build', $Project, '-c', 'Release', '--no-restore', '-v', 'minimal') + $properties)

$Frameworks = @('net48', 'netcoreapp3.1', 'net5.0', 'net6.0', 'net7.0', 'net8.0', 'net9.0', 'net10.0')
$installed = & dotnet --list-runtimes
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect installed .NET runtimes.' }
$records = @()
foreach ($framework in $Frameworks) {
    $arguments = @('--report', (Join-Path $SmokeRoot "$framework.json"))
    if ($ModelPath -and $framework -in @('net48', 'netcoreapp3.1', 'net8.0')) {
        $arguments += @('--model', $ModelPath)
    }
    if ($framework -eq 'net48') {
        $program = Join-Path $SmokeRoot "bin/Release/$framework/DotnetCompat.exe"
        Invoke-Checked $program $arguments
    }
    else {
        $minor = $framework -replace '^net(coreapp)?', ''
        if (!($installed -match "^Microsoft.NETCore.App $([regex]::Escape($minor))\.")) {
            $records += @{ target = $framework; compiled = $true; executed = $false; reason = 'Runtime is not installed.' }
            continue
        }
        $program = Join-Path $SmokeRoot "bin/Release/$framework/win-x64/DotnetCompat.dll"
        Invoke-Checked dotnet (@($program) + $arguments)
    }
    $record = Get-Content -LiteralPath (Join-Path $SmokeRoot "$framework.json") -Raw | ConvertFrom-Json
    $records += $record
}

# Framework x86 can execute on an x64 Windows host; ARM64 needs its own host.
$properties32 = @(
    "-p:TurboJevPackageVersion=$Version",
    "-p:RestorePackagesPath=$SmokeRoot/packages",
    "-p:BaseIntermediateOutputPath=$SmokeRoot/x86-obj/",
    "-p:BaseOutputPath=$SmokeRoot/x86/",
    '-p:TargetFrameworks=net48',
    '-p:PlatformTarget=x86',
    '-p:RuntimeIdentifier=win-x86'
)
Invoke-Checked dotnet (@('restore', $Project, '--configfile', $Config, '-v', 'minimal') + $properties32)
Invoke-Checked dotnet (@('build', $Project, '-f', 'net48', '-c', 'Release', '--no-restore', '-v', 'minimal') + $properties32)
$report32 = Join-Path $SmokeRoot 'net48-x86.json'
Invoke-Checked (Join-Path $SmokeRoot 'x86/Release/net48/win-x86/DotnetCompat.exe') @('--report', $report32)
$records += (Get-Content -LiteralPath $report32 -Raw | ConvertFrom-Json)

@{ package_version = $Version; package_sha256 = (Get-FileHash -LiteralPath $PackagePath -Algorithm SHA256).Hash.ToLowerInvariant(); records = $records } |
    ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $SmokeRoot 'report.json')
Write-Host "Installed NuGet framework compatibility passed. Report: $SmokeRoot/report.json"
