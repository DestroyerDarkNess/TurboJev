# Optional portable ARM64 tools extracted from Microsoft's signed VS packages.
# Normal installations are detected by Cargo without this helper.
$portableRoot = Join-Path $PSScriptRoot '../artifacts/msvc-arm64/VC/Tools/MSVC'
if (Test-Path $portableRoot) {
    $toolset = Get-ChildItem $portableRoot -Directory | Sort-Object Name -Descending | Select-Object -First 1
    $sdkRoot = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits/10/Lib'
    $sdk = Get-ChildItem $sdkRoot -Directory | Sort-Object Name -Descending | Where-Object { Test-Path (Join-Path $_.FullName 'um/arm64/kernel32.lib') } | Select-Object -First 1
    if (-not $sdk) { throw 'Windows SDK ARM64 libraries are required' }
    $armBin = Join-Path $toolset.FullName 'bin/Hostx64/arm64'
    $linker = Join-Path $armBin 'link.exe'
    $libPaths = @((Join-Path $toolset.FullName 'lib/arm64'), (Join-Path $toolset.FullName 'lib/onecore/arm64'), (Join-Path $sdk.FullName 'ucrt/arm64'), (Join-Path $sdk.FullName 'um/arm64'))
    $launcher = Join-Path $PSScriptRoot '../artifacts/msvc-arm64/link-arm64.cmd'
    @('@echo off', 'setlocal', ('set "LIB=' + ($libPaths -join ';') + '"'), ('"' + $linker + '" %*'), 'exit /b %errorlevel%') | Set-Content $launcher
    $env:CARGO_TARGET_AARCH64_PC_WINDOWS_MSVC_LINKER = (Resolve-Path $launcher).Path
    $env:TURBOJEV_ARM64_LIB_EXE = Join-Path $armBin 'lib.exe'
}
