#!/usr/bin/env bash
set -euo pipefail

: "${RID:?RID must select a Linux release target}"
export PATH="/opt/python/cp312-cp312/bin:$HOME/.cargo/bin:$PATH"
export CARGO_CACHE_AUTO_CLEAN_FREQUENCY=never
git config --global --add safe.directory /work
git config --global --add safe.directory /work/.cache/real-validation/llama.cpp
python -m pip install maturin==1.9.6 cmake ninja auditwheel
if ! command -v cargo >/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |
    sh -s -- -y --profile minimal --default-toolchain stable
fi

# Node publishes official Linux x64 and ARM64 binaries. The Linux x86 addon
# is architecture-checked; no unofficial Node interpreter is substituted.
case "$RID" in
  linux-x64) node_arch=x64 ;;
  linux-arm64) node_arch=arm64 ;;
  linux-x86) node_arch= ;;
  *) echo "Unsupported Linux target: $RID" >&2; exit 1 ;;
esac
if [[ -n "$node_arch" ]]; then
  node_archive="node-v24.18.0-linux-$node_arch.tar.xz"
  curl --fail --location "https://nodejs.org/dist/v24.18.0/$node_archive" -o "/tmp/$node_archive"
  tar -xJf "/tmp/$node_archive" -C /tmp
  export PATH="/tmp/node-v24.18.0-linux-$node_arch/bin:$PATH"
fi
python scripts/build-native-unix.py "$RID"
