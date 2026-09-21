#!/usr/bin/env sh
set -eu

# Developer convenience only. TurboJev itself never depends on this script.
if command -v cargo >/dev/null 2>&1 && command -v rustc >/dev/null 2>&1; then
  cargo --version
  rustc --version
  exit 0
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required to bootstrap rustup" >&2
  exit 1
fi

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
. "$HOME/.cargo/env"
rustup toolchain install 1.82.0 --profile minimal --component rustfmt --component clippy
rustup target add wasm32-unknown-unknown --toolchain 1.82.0
cargo +1.82.0 --version
rustc +1.82.0 --version
