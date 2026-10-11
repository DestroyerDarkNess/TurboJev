#!/usr/bin/env bash
# Linux fallback for building macOS artifacts. Native execution is a separate gate.
set -euo pipefail

: "${MACOS_SDK:?Set MACOS_SDK to an extracted macOS SDK}"
rid="${1:?Select osx-x64 or osx-arm64}"
case "$rid" in
  osx-x64) arch=x86_64; triple=x86_64-apple-darwin ;;
  osx-arm64) arch=arm64; triple=aarch64-apple-darwin ;;
  *) echo "Unsupported macOS target: $rid" >&2; exit 1 ;;
esac
root="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.cargo/bin:$PATH"
# Proc macros and build scripts also need the Linux host standard library.
(cd "$root" && rustup target add x86_64-unknown-linux-gnu "$triple")
out="$root/artifacts/unix/$rid"
mkdir -p "$out/native" "$out/node" "$out/wheels" "$out/toolchain"
source_cache="$HOME/.cache/turbojev-toolchains/source"
mkdir -p "$source_cache"
if [[ ! -f "$source_cache/llama.cpp/CMakeLists.txt" ]]; then
  cp -a "$root/.cache/real-validation/llama.cpp" "$source_cache/llama.cpp"
fi
mkdir -p "$source_cache/native"
cp -a "$root/crates/turbojev-runtime-llamacpp/native/." "$source_cache/native/"

cat > "$out/toolchain/clang" <<EOF
#!/usr/bin/env bash
exec clang --target=$arch-apple-darwin -fuse-ld=lld -isysroot "$MACOS_SDK" -mmacosx-version-min=11.0 "\$@"
EOF
chmod +x "$out/toolchain/clang"
cat > "$out/toolchain/after-project.cmake" <<'EOF'
# Linux has no sw_vers. Supply Darwin's supported rpath flags after platform setup.
set(CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG "-Wl,-rpath,")
set(CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG "-Wl,-rpath,")
EOF
cat > "$out/toolchain/darwin.cmake" <<EOF
set(CMAKE_SYSTEM_NAME Darwin)
set(CMAKE_SYSTEM_VERSION 20.0)
set(CMAKE_SYSTEM_PROCESSOR $arch)
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_ASM_COMPILER clang)
set(CMAKE_C_COMPILER_TARGET $arch-apple-darwin)
set(CMAKE_CXX_COMPILER_TARGET $arch-apple-darwin)
set(CMAKE_ASM_FLAGS_INIT "--target=$arch-apple-darwin -isysroot $MACOS_SDK")
set(CMAKE_CXX_FLAGS_INIT "-stdlib=libc++")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-fuse-ld=lld")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "-fuse-ld=lld")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "-fuse-ld=lld")
set(CMAKE_OSX_SYSROOT "$MACOS_SDK")
set(CMAKE_OSX_ARCHITECTURES $arch)
set(CMAKE_OSX_DEPLOYMENT_TARGET 11.0)
set(CMAKE_AR /usr/bin/llvm-ar-18)
set(CMAKE_RANLIB /usr/bin/llvm-ranlib-18)
set(CMAKE_INSTALL_NAME_TOOL /usr/bin/llvm-install-name-tool-18)
EOF

build="$HOME/.cache/turbojev-toolchains/build-$rid"
cmake -S "$source_cache/native" -B "$build" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$out/toolchain/darwin.cmake" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PROJECT_INCLUDE="$out/toolchain/after-project.cmake" \
  -DCMAKE_C_FLAGS= -DCMAKE_CXX_FLAGS=-stdlib=libc++ \
  -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld \
  -DLLAMA_CPP_SOURCE_DIR="$source_cache/llama.cpp" \
  -DGGML_OPENMP=OFF -DGGML_NATIVE=OFF -DGGML_VULKAN=OFF \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_BLAS=OFF \
  -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF
cmake --build "$build" --target turbojev_llama_shim --parallel 4
# Do not retain a BLAS backend from a previous build with a newer Accelerate ABI.
find "$out/native" "$out/node" -maxdepth 1 -name 'libggml-blas*.dylib' -delete
find "$build" -name '*.dylib' ! -name 'libggml-blas*.dylib' -exec cp -L '{}' "$out/native/" \;

export PATH="$HOME/.cargo/bin:$PATH"
if [[ ! -x "$HOME/.cache/turbojev-toolchains/python/bin/python" ]]; then
  python3 -m venv "$HOME/.cache/turbojev-toolchains/python"
  "$HOME/.cache/turbojev-toolchains/python/bin/python" -m pip install maturin==1.9.6
fi
export PATH="$HOME/.cache/turbojev-toolchains/python/bin:$PATH"
export TURBOJEV_LLAMA_SHIM_LIB_DIR="$out/native"
export MACOSX_DEPLOYMENT_TARGET=11.0
export CARGO_TARGET_DIR="$HOME/.cache/turbojev-cross-target"
export RUSTFLAGS="-C linker=$out/toolchain/clang -C link-arg=-Wl,-rpath,@loader_path"
export PYO3_CONFIG_FILE="$out/toolchain/pyo3.txt"
printf '%s\n' 'implementation=CPython' 'version=3.9' 'shared=true' 'abi3=true' \
  'lib_name=python3.9' 'pointer_width=64' > "$PYO3_CONFIG_FILE"
cd "$root"
cargo build --locked --release --target "$triple" -p turbojev-capi -p turbojev-node \
  --features turbojev-capi/llamacpp,turbojev-node/llamacpp
cp "$CARGO_TARGET_DIR/$triple/release/libturbojev_capi.dylib" "$out/native/"
cp "$out/native/"*.dylib "$out/node/"
cp "$CARGO_TARGET_DIR/$triple/release/libturbojev_node.dylib" "$out/node/turbojev.node"
python3 scripts/package-macos-cross.py "$rid"
