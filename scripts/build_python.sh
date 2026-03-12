#!/usr/bin/env bash
# Build the Python extension and install it into the python/ignition/ package directory.
#
# Usage:
#   ./scripts/build_python.sh          # release build (fast)
#   ./scripts/build_python.sh --debug  # debug build (for development)
#
# Workaround: maturin detects WSL as cross-compilation and looks for
# /usr/bin/python (which doesn't exist on Ubuntu). We build with cargo directly
# and install the .so manually. The PYO3_PYTHON env var is set in
# .cargo/config.toml so cargo finds the right interpreter.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Source Rust toolchain
source "$HOME/.cargo/env" 2>/dev/null || true

PROFILE="release"
CARGO_FLAG="--release"
if [[ "${1:-}" == "--debug" ]]; then
    PROFILE="debug"
    CARGO_FLAG=""
fi

echo "▶ Building ignition extension (${PROFILE})..."
cargo build --lib --features python $CARGO_FLAG \
    --manifest-path "${PROJECT_DIR}/Cargo.toml"

# Determine the Python extension suffix (.cpython-312-x86_64-linux-gnu.so)
EXT=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

SRC="${PROJECT_DIR}/target/${PROFILE}/libignition.so"
DST="${PROJECT_DIR}/python/ignition/_ignition${EXT}"

echo "▶ Installing ${SRC} → ${DST}"
cp "${SRC}" "${DST}"

echo "✓ Done. Set PYTHONPATH=${PROJECT_DIR}/python to use the extension."
