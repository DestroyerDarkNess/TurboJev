param(
    [Parameter(Mandatory = $true)][string]$PackagePath,
    [Parameter(Mandatory = $true)][string]$ModelPath
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PackagePath = (Resolve-Path -LiteralPath $PackagePath).Path
$ModelPath = (Resolve-Path -LiteralPath $ModelPath).Path
$Version = [regex]::Match([IO.Path]::GetFileName($PackagePath), '^TurboJev\.(.+)\.nupkg$').Groups[1].Value
if ([string]::IsNullOrWhiteSpace($Version)) { throw "Cannot determine package version from $PackagePath" }

$SmokeRoot = Join-Path $Root "artifacts/nuget-smoke"
New-Item -ItemType Directory -Force -Path $SmokeRoot | Out-Null
$Project = Join-Path $SmokeRoot "TurboJev.PackageSmoke.csproj"
$Program = Join-Path $SmokeRoot "Program.cs"
$NuGetConfig = Join-Path $SmokeRoot "NuGet.Config"
$PackageSource = Split-Path -Parent $PackagePath
$csproj = @"
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net8.0</TargetFramework><RuntimeIdentifier>win-x64</RuntimeIdentifier><ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable></PropertyGroup>
  <ItemGroup><PackageReference Include="TurboJev" Version="$Version" /></ItemGroup>
</Project>
"@
$source = @'
using System.Text.Json;
using TurboJev;
if (args.Length != 1) throw new ArgumentException("Pass the pinned GGUF path.");
await using var engine = await TurboJevEngine.LoadAsync(new { model = args[0], device="cpu", nGpuLayers = 0, nThreads = 4,nCtx=8192,nSeqMax=1 });
var request = File.ReadAllText(Path.Combine(Environment.GetEnvironmentVariable("TURBOJEV_SOURCE_ROOT")!, "tests/e2e/request.json"));
var output = await engine.EvaluateJsonAsync(request);
using var response = JsonDocument.Parse(output);
if (response.RootElement.GetProperty("_turbojev").GetProperty("execution").GetProperty("prefix_reuse").GetString() != "none") throw new Exception("One sequence cannot retain a shared prefix and a separate suffix");
foreach (var key in new[] { "route", "urgent", "severity" })
    if (!response.RootElement.GetProperty("answers").TryGetProperty(key, out _)) throw new Exception($"Missing typed answer {key}");
using var generated = JsonDocument.Parse(await engine.GenerateJsonAsync("""
{"context":"There are exactly seven items.","questions":{"count":{"type":"integer","instructions":"How many items are stated?"},"seen":{"type":"boolean","depends_on":["count"],"instructions":"Was an item count provided?"}}}
"""));
if (!generated.RootElement.GetProperty("result").GetProperty("count").TryGetInt64(out _)) throw new Exception("Missing generated integer");
if (generated.RootElement.GetProperty("result").GetProperty("seen").ValueKind is not (JsonValueKind.True or JsonValueKind.False)) throw new Exception("Missing dependent boolean");
if (generated.RootElement.GetProperty("usage").GetProperty("generated_tokens").GetInt32()==0) throw new Exception("No actual generation was performed");
await using var balanced = await TurboJevEngine.LoadAsync(new {
    model = "fixture://balanced-package-gate", execution = new { orderRobustness = "balanced" },
    fixtureLogits = new { route = new[] { 0.0, 0.0, 0.0 } }
});
using var corrected = JsonDocument.Parse(await balanced.EvaluateJsonAsync("""
{"state":"facts","questions":{"route":{"type":"choice","instructions":"Choose","criteria":{"z":"Z","a":"A","b":"B"}}}}
"""));
if (corrected.RootElement.GetProperty("answers").GetProperty("route").GetProperty("choice").GetString() != "a") throw new Exception("Exact tie did not use semantic keys");
if (corrected.RootElement.GetProperty("_turbojev").GetProperty("order_robustness").GetString() != "balanced") throw new Exception("Balanced policy was ignored");
if (corrected.RootElement.GetProperty("_turbojev").GetProperty("order_evaluations").GetInt32() != 6) throw new Exception("Unexpected balanced pass count");
Console.WriteLine("Built NuGet package real-model evaluate/generate and balanced contract smoke tests passed.");
'@
Set-Content -LiteralPath $Project -Value $csproj -NoNewline
Set-Content -LiteralPath $Program -Value $source -NoNewline
$escapedSource = [Security.SecurityElement]::Escape($PackageSource)
$config = @"
<?xml version="1.0" encoding="utf-8"?>
<configuration><packageSources><clear/><add key="TurboJev smoke package" value="$escapedSource"/><add key="nuget.org" value="https://api.nuget.org/v3/index.json"/></packageSources></configuration>
"@
Set-Content -LiteralPath $NuGetConfig -Value $config -NoNewline
$globalPackagesInfo = & dotnet nuget locals global-packages --list
if ($LASTEXITCODE -ne 0 -or $globalPackagesInfo -notmatch '^global-packages:\s*(.+)$') {
    throw "Cannot locate the NuGet global package cache."
}
$globalPackages = [IO.Path]::GetFullPath($matches[1].Trim())
$cachedPackage = [IO.Path]::GetFullPath((Join-Path $globalPackages "turbojev/$Version"))
if (-not $cachedPackage.StartsWith($globalPackages.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to clear an unexpected NuGet cache path: $cachedPackage"
}
if (Test-Path -LiteralPath $cachedPackage) {
    Remove-Item -LiteralPath $cachedPackage -Recurse -Force
}
$env:TURBOJEV_SOURCE_ROOT = $Root
try {
    dotnet restore $Project --configfile $NuGetConfig --no-cache -r win-x64
    if ($LASTEXITCODE -ne 0) { throw "NuGet package smoke restore failed." }
    dotnet run --project $Project --configuration Release --no-restore -r win-x64 -- $ModelPath
    if ($LASTEXITCODE -ne 0) { throw "NuGet package real-model smoke test failed." }
} finally {
    Remove-Item Env:TURBOJEV_SOURCE_ROOT -ErrorAction SilentlyContinue
}
