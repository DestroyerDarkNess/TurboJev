# Native releases

Version 0.34.0 uses one matrix across npm, PyPI and NuGet. The inference engine
and its shared library dependencies ship inside each native package. No separate
llama.cpp executable is required.

| Runtime directory | npm package | Python wheel platform | GPU backend |
| --- | --- | --- | --- |
| win-x86 | @turbojev/node-win32-ia32 | win32 | CPU |
| win-x64 | @turbojev/node-win32-x64 | win_amd64 | Vulkan |
| win-arm64 | @turbojev/node-win32-arm64 | win_arm64 | CPU |
| linux-x86 | @turbojev/node-linux-ia32 | manylinux_2_28_i686 | CPU |
| linux-x64 | @turbojev/node-linux-x64 | manylinux_2_28_x86_64 | CPU |
| linux-arm64 | @turbojev/node-linux-arm64 | manylinux_2_28_aarch64 | CPU |
| osx-x64 | @turbojev/node-darwin-x64 | macosx_11_0_x86_64 | Metal |
| osx-arm64 | @turbojev/node-darwin-arm64 | macosx_11_0_arm64 | Metal |

Linux requires glibc 2.28 or newer. Alpine/musl is not covered by these binaries.
macOS requires version 11 or newer. Modern macOS has no 32-bit application
runtime. Python requires normal CPython 3.9+; Node requires 18+. The Linux x86
addon requires a compatible Node interpreter, which current official Node Linux
downloads do not provide. A native library does not make an unsupported host
runtime available: .NET's own OS/architecture matrix still applies.

## Local builds

The authoritative target list is `scripts/native-targets.json`. No Actions run
is required. On Windows, prepare its three targets and browser packages:

```powershell
pwsh -NoProfile -File scripts/publish-local.ps1 -Target all -Yes -BuildOnly
```

Build each Linux target inside its corresponding manylinux image. For example:

```sh
docker run --rm -v "$PWD:/work" -w /work -e RID=linux-x64 \
  quay.io/pypa/manylinux_2_28_x86_64 bash scripts/build-manylinux.sh
```

Use `manylinux_2_28_i686` with `linux-x86` and
`manylinux_2_28_aarch64` with `linux-arm64`. An ARM64 host or registered QEMU
emulator is required for the ARM64 container. The build runs Python and, where
available, Node inference against a real GGUF after hiding original CMake
libraries and clearing loader environment overrides.

To avoid emulating the ARM64 compiler, LLVM 18 can cross-build against a paused
ARM64 manylinux container's filesystem. Set `MANYLINUX_SYSROOT` to its mounted
root (for example `/proc/<container-pid>/root`) and run
`scripts/cross-build-linux-arm64.sh` on Linux x64. Afterward, execute
`python scripts/validate-unix-package.py linux-arm64` inside an ARM64 container
or matching host. Cross-build evidence alone does not qualify Linux publication.

On a matching Mac, install CMake, Rust, Python, maturin and Node, then run:

```sh
python scripts/build-native-unix.py osx-arm64
# Intel Mac: python scripts/build-native-unix.py osx-x64
```

Linux can cross-build macOS with Clang/LLVM 18, lld, Ninja, Python venv support,
the Rust Apple targets and an extracted macOS SDK:

```sh
export MACOS_SDK=/path/to/MacOSX.sdk
rustup target add aarch64-apple-darwin x86_64-apple-darwin
bash scripts/cross-build-macos.sh osx-arm64
bash scripts/cross-build-macos.sh osx-x64
```

The output is `artifacts/unix/<rid>` with `native`, `node`, `wheels` and
`build-evidence.json`. Cross-built Mach-O libraries are checked for architecture,
ABI exports, bundled dependencies, relocatable paths, OS deployment target and
linker code signatures (required for ARM64).
These checks do not claim execution on macOS or GPU hardware.

## Publish

Keep all Unix artifacts in their output directories, then run the Windows
publisher. It checks all eight native targets before upload:

```powershell
pwsh -NoProfile -File scripts/publish-local.ps1 -Target all -Yes
python scripts/verify-published-release.py --version 0.34.0
```

The publisher verifies build evidence and library hashes. Linux requires a real
Python inference check. macOS can qualify with either native execution or the
explicit cross-build link checks. Cargo publishes portable source rather than
one binary per target; the browser packages publish WebAssembly once.

## 0.34.0 verification status

The current registry report is
[published-release-0.34.0.json](validation/published-release-0.34.0.json). npm's
13 packages, including all eight native packages, passed tarball integrity,
architecture and library-hash checks. All eight PyPI wheels matched their
published SHA-256 hashes, all 27 required crates.io packages are present, and
NuGet 0.34.0 passed public package and payload checks, including all eight
native runtime directories. The linked report records the registry evidence.

Real GGUF smoke runs passed on Windows x64 (Python, Node and .NET), Linux x64
(Python and Node), Linux x86 (Python), and Linux ARM64 under QEMU on Linux x64
(Python and Node). macOS x64/ARM64 passed cross-build, package and linker checks;
neither was executed on a Mac. Windows x86/ARM64 passed package and architecture
checks but was not run on matching hardware. These checks establish loading and
typed response behavior, not model accuracy or GPU execution on every target.
