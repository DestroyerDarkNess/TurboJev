$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$output = Join-Path $repoRoot '.artifacts/wasm-tests'

Push-Location $repoRoot
try {
    cargo build -p turbojev-wasm --target wasm32-unknown-unknown --release --locked
    if ($LASTEXITCODE -ne 0) {
        throw 'WASM compilation failed. Install the target with rustup target add wasm32-unknown-unknown.'
    }

    $bindgen = Join-Path $repoRoot '.cache/wasm-tools/bin/wasm-bindgen.exe'
    if (-not (Test-Path -LiteralPath $bindgen)) {
        $bindgen = (Get-Command wasm-bindgen -ErrorAction SilentlyContinue).Source
    }
    if (-not $bindgen) {
        throw 'Install wasm-bindgen-cli 0.2.100 before preparing the tests.'
    }

    $metadata = cargo metadata --locked --no-deps --format-version 1 | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not read the Cargo target directory.'
    }
    $wasm = Join-Path $metadata.target_directory 'wasm32-unknown-unknown/release/turbojev_wasm.wasm'
    & $bindgen --target web --out-dir $output $wasm
    if ($LASTEXITCODE -ne 0) {
        throw 'wasm-bindgen failed.'
    }

    '{"type":"module"}' | Set-Content -LiteralPath (Join-Path $output 'package.json') -Encoding utf8
    Write-Output "WASM test artifacts ready: $output"
    Write-Output 'Serve browser tests: python scripts/serve-browser-tests.py --port 4176'
} finally {
    Pop-Location
}
