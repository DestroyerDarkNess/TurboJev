#!/usr/bin/env bash
# Cross-build against an ARM64 manylinux container's filesystem, without emulating compilers.
set -euo pipefail
: "${MANYLINUX_SYSROOT:?Set the mounted ARM64 manylinux filesystem path}"
root="$(cd "$(dirname "$0")/.." && pwd)"
tools="$HOME/.cache/turbojev-linux-arm64"
mkdir -p "$tools"
gcc_dir="$MANYLINUX_SYSROOT/opt/rh/gcc-toolset-14/root/usr/lib/gcc/aarch64-redhat-linux/14"
for compiler in clang clang++; do
  cat > "$tools/$compiler" <<EOF
#!/usr/bin/env bash
exec /usr/bin/$compiler --target=aarch64-unknown-linux-gnu --sysroot="$MANYLINUX_SYSROOT" --gcc-install-dir="$gcc_dir" -fuse-ld=lld -Wno-unused-command-line-argument "\$@"
EOF
  chmod +x "$tools/$compiler"
done
cat > "$tools/toolchain.cmake" <<EOF
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER "$tools/clang")
set(CMAKE_CXX_COMPILER "$tools/clang++")
set(CMAKE_AR /usr/bin/llvm-ar-18)
set(CMAKE_RANLIB /usr/bin/llvm-ranlib-18)
EOF
python3 -m venv "$tools/python"
"$tools/python/bin/python" -m pip install maturin==1.9.6 auditwheel
export PATH="$tools/python/bin:$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR="$HOME/.cache/turbojev-cross-target"
export RUSTFLAGS="-C linker=$tools/clang -C link-arg=-Wl,-rpath,\$ORIGIN"
export TURBOJEV_CMAKE_TOOLCHAIN="$tools/toolchain.cmake"
export TURBOJEV_NATIVE_BUILD_DIR="$tools/cmake"
export TURBOJEV_LLAMA_SOURCE_DIR="$HOME/.cache/turbojev-toolchains/source/llama.cpp"
export PYO3_CONFIG_FILE="$tools/pyo3.txt"
printf '%s\n' 'implementation=CPython' 'version=3.9' 'shared=true' 'abi3=true' \
  'lib_name=python3.9' 'pointer_width=64' > "$PYO3_CONFIG_FILE"
cd "$root"
python scripts/build-native-unix.py linux-arm64 --cross-build
