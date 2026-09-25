$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    cargo build -p turbojev-wasm --target wasm32-unknown-unknown --release --locked
    if ($LASTEXITCODE -ne 0) { throw 'WASM compilation failed. Install the target with rustup target add wasm32-unknown-unknown.' }
    $bindgen = Join-Path $repoRoot '.cache/wasm-tools/bin/wasm-bindgen.exe'
    if (-not (Test-Path -LiteralPath $bindgen)) {
        $bindgen = (Get-Command wasm-bindgen -ErrorAction SilentlyContinue).Source
    }
    if (-not $bindgen) { throw 'Install wasm-bindgen-cli 0.2.100 or run python samples/build.py web first.' }
    $metadata = cargo metadata --no-deps --format-version 1 | ConvertFrom-Json
    $wasm = Join-Path $metadata.target_directory 'wasm32-unknown-unknown/release/turbojev_wasm.wasm'
    & $bindgen --target web --out-dir website/vendor/wasm $wasm
    if ($LASTEXITCODE -ne 0) { throw 'wasm-bindgen failed.' }
    if (-not (Test-Path 'samples/TurboJev-js-native/node_modules/esbuild')) {
        npm ci --prefix samples/TurboJev-js-native
        if ($LASTEXITCODE -ne 0) { throw 'Browser dependencies could not be installed.' }
    }
    node website/build.mjs
    if ($LASTEXITCODE -ne 0) { throw 'Website build failed.' }
    Write-Output 'Website, documentation and current WASM runtime are ready.'
} finally { Pop-Location }
