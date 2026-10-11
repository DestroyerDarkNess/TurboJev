# Local package publishing

Run `pwsh -NoProfile -File scripts/publish-local.ps1` from the repository root.
Choose npm, PyPI, NuGet, crates.io, or all, then enter `PUBLISH`.
For non-interactive operation use `-Target all -Yes`, or select an individual
registry with `-Target npm`, `python`, `nuget`, or `cargo`.

Store credentials in the ignored root `.env` file using `.env.example` as the
template. Never commit credentials. The local publisher needs API credentials;
GitHub trusted-publisher configurations alone do not authenticate local uploads.

## Prerequisites

- Windows with Rust, Python 3.11 or newer, Node.js/npm, .NET 10 SDK, .NET Framework 4.8 runtime, CMake 3.24+, and Git.
- Visual Studio C++ x86/x64 and ARM64 build tools, Windows SDK, and LLVM tools.
- npm membership and token write access to the `@turbojev` organization.
- A verified email address on crates.io and publishing permission.
- PyPI and NuGet tokens permitting initial packages and new versions.

## Packages and architectures

- npm publishes the web binding and browser runtime adapters once, then eight
  native Node packages for Windows/Linux ia32/x64/arm64 and macOS x64/arm64,
  followed by the Node wrapper. Browser
  demos load the published llama.cpp runtime from jsDelivr; they do not copy it
  from the repository source.
- PyPI publishes eight Windows/Linux/macOS wheels using the CPython
  stable ABI (Python 3.9+). Publishing tools use `artifacts/publish-tools` so they
  do not modify globally installed Python packages.
- NuGet publishes one package containing all eight desktop runtime directories
  from `scripts/native-targets.json`, plus managed assemblies for `netstandard2.0`,
  `net48` and `net8.0`. These managed compatibility targets are included starting
  in version 0.33.1.
- crates.io publishes source packages once. The facade's internal dependencies
  are published first. Cargo verifies each packaged crate before upload.

The publisher builds the pinned llama.cpp revision once per Windows target and
enables the `llamacpp` feature for each native binding. The shim and llama.cpp
runtime DLLs are bundled in npm packages, Python wheels, and NuGet runtime
folders. Prepare Linux/macOS artifacts first using the local commands in
[NATIVE_RELEASES.md](NATIVE_RELEASES.md). The publisher requires their evidence
and validates hashes before upload. `-BuildOnly` builds the Windows/browser
artifacts without uploading and can run before Unix artifacts are ready.
Cross-compilation does not execute ARM64 binaries on an x64 host. The .NET compatibility gate also
executes the Framework consumer as x86 on Windows x64.

Before uploading NuGet, the publisher installs the actual packed artifact into
isolated consumers targeting Framework 4.8, .NET Core 3.1 and .NET 5â€“10.
It compiles every target, executes those with an installed runtime, and tests
real GGUF inference in Framework 4.8, Core 3.1 and .NET 8. Missing legacy
runtimes are reported as compile-only coverage. .NET 8 must be installed for
the existing native package smoke test. To repeat the compatibility gate:

```powershell
pwsh -NoProfile -File scripts/smoke-dotnet-compat.ps1 `
  -PackagePath artifacts/nuget/TurboJev.VERSION.nupkg `
  -ModelPath path/to/model.gguf
```

## Retry after a failure

Cargo registry verification uses the installed `stable` Rust toolchain because
dependency resolution during packaging can select crates using newer Rust
editions than the workspace's pinned toolchain. To select another installed
toolchain, set `$env:TURBOJEV_PUBLISH_TOOLCHAIN = '1.97.1'` before publishing.
This changes the publication verifier, not the project's source toolchain.

Select only the failed registry. Existing crate versions, npm versions and Python
wheels are skipped; NuGet uses `--skip-duplicate`. A failed registry
does not prevent the `all` option from attempting the remaining registries.

Compilation alone is not a successful publication. Check the final registry
response and confirm that the expected version and platform assets are visible.

```powershell
python scripts/verify-published-release.py
python scripts/verify-published-release.py --registry npm --version 0.30.1
```

The verifier uses public registry endpoints without credentials. It checks all
thirteen npm packages, the complete published Cargo dependency closure, all eight
Python wheel platforms, and NuGet's published checksum. When local
release artifacts are present, it also compares wheel hashes and checks the
NuGet package's native assets across all eight runtime directories. Its JSON report is saved
under `artifacts/`; missing or unavailable packages make it fail.
