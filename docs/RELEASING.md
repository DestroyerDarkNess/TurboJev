# Releasing TurboJev

TurboJev releases are tag-driven. The release workflow checks that the tag
version matches the Rust workspace, JavaScript packages, Python project, and
.NET package. A release is started by pushing a tag such as `v0.28.3`.

## One-time setup

Create or reserve these package names before the first publish:

- crates.io: the Rust crates that are intended to be public
- npm: `@turbojev/web` and `@turbojev/runtime-transformersjs`
- PyPI: `turbojev`
- NuGet: `TurboJev`

Configure the `release` environment in the GitHub repository. Require a
reviewer for that environment if releases should be manually approved.

Configure trusted publishing using the official registry guides for [npm](https://docs.npmjs.com/trusted-publishers/), [PyPI](https://docs.pypi.org/trusted-publishers/), and [NuGet](https://learn.microsoft.com/en-us/nuget/nuget-org/trusted-publishing) with:

- repository owner: `DestroyerDarkNess`
- repository: `TurboJev`
- workflow: `.github/workflows/release.yml`
- environment: `release`

The workflow requests the GitHub OIDC identity token. No long-lived npm, PyPI,
or NuGet token belongs in the repository. Cargo currently uses the protected
`CARGO_REGISTRY_TOKEN` repository or environment secret because crates.io does
not provide the same trusted-publisher flow. Follow the [Cargo publishing
reference](https://doc.rust-lang.org/cargo/reference/publishing.html) when the
workspace dependency metadata is ready for registry publication.

## Publishing a release

1. Merge the changes and run the real-model workflow on the commit.
2. Update the workspace version and every binding manifest to the same value.
3. Run `python scripts/release_preflight.py --tag vX.Y.Z` locally.
4. Push the tag: `git push origin vX.Y.Z`.
5. The release workflow publishes the web packages and Python wheels, then
   attaches a source archive to the GitHub release.

The web packages and Python wheel are the first automated registry targets.
Node, .NET, and llama.cpp-backed Python packages need platform-native assets in
the release artifact before they can be published as install-and-run packages.
The workflow keeps those artifacts attached to the GitHub release while that
matrix is completed; it does not publish a package that would install without
its native library.

## Architecture policy

Pure Rust crates and browser packages are architecture-independent. Native
bindings must be built and tested for each supported runtime identifier, then
placed in the package's native asset directory:

- Node: one optional package per `process.platform` and `process.arch`
- Python: one wheel per Python ABI and platform
- .NET: `runtimes/<rid>/native/`

Windows x64, Linux x64, and macOS x64 are the initial release artifacts. ARM64
and 32-bit Windows require dedicated runners or cross-compilers and should be
added to the matrix only after a real-model smoke test passes for that target.

## Dry runs

Use these checks before creating a tag:

```text
python scripts/release_preflight.py --tag v0.28.3
cargo metadata --locked
cargo publish --dry-run -p <crate>
npm pack --dry-run
python -m build
dotnet pack --configuration Release
```

Cargo packages are immutable once published. Publish dependency crates in
topological order and never reuse a released version for a changed artifact.
