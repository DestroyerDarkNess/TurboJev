# Local package publishing

Run `pwsh -NoProfile -File scripts/publish-local.ps1` from the repository root.
Choose npm, PyPI, NuGet, crates.io, or all, then enter `PUBLISH`.
For non-interactive operation use `-Target all -Yes`, or select an individual
registry with `-Target npm`, `python`, `nuget`, or `cargo`.

Store credentials in the ignored root `.env` file using `.env.example` as the
template. Never commit credentials. The local publisher needs API credentials;
GitHub trusted-publisher configurations alone do not authenticate local uploads.

## Prerequisites

- Windows with Rust, Python 3.11 or newer, Node.js/npm, .NET 8, CMake 3.24+, and Git.
- Visual Studio C++ x86/x64 and ARM64 build tools, Windows SDK, and LLVM tools.
- npm membership and token write access to the `@turbojev` organization.
- A verified email address on crates.io and publishing permission.
- PyPI and NuGet tokens permitting initial packages and new versions.

## Packages and architectures

- npm publishes web packages once, then Windows native Node packages for ia32,
  x64, and arm64 and the Node wrapper.
- PyPI publishes three Windows wheels: x86, x64, and ARM64, using the CPython
  stable ABI (Python 3.9+). Publishing tools use `artifacts/publish-tools` so they
  do not modify globally installed Python packages.
- NuGet publishes one package containing native Windows runtime assets for
  win-x86, win-x64, and win-arm64.
- crates.io publishes source packages once. The facade's internal dependencies
  are published first. Cargo verifies each packaged crate before upload.

The publisher builds the pinned llama.cpp revision once per Windows target and
enables the `llamacpp` feature for each native binding. The shim and llama.cpp
runtime DLLs are bundled in npm packages, Python wheels, and NuGet runtime
folders. This Windows matrix does not produce Linux or macOS binaries and does
not execute x86 or ARM64 binaries on an x64 host.

## Retry after a failure

Select only the failed registry. Existing crate versions and Python wheels are
skipped; NuGet uses `--skip-duplicate`. npm rejects an existing version, so check
which packages were successfully uploaded before retrying. A failed registry
does not prevent the `all` option from attempting the remaining registries.

Compilation alone is not a successful publication. Check the final registry
response and confirm that the expected version and platform assets are visible.
