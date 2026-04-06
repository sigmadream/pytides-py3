#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="${UV_CACHE_DIR:-${TMPDIR:-/tmp}/uv-cache}"
SUPPORTED_VERSIONS=(3.10 3.11 3.12 3.13)
EXPERIMENTAL_VERSIONS=(3.14)

MODE="${1:---all}"

case "$MODE" in
  --supported-only)
    RUN_SUPPORTED=1
    RUN_EXPERIMENTAL=0
    ;;
  --experimental-only)
    RUN_SUPPORTED=0
    RUN_EXPERIMENTAL=1
    ;;
  --all)
    RUN_SUPPORTED=1
    RUN_EXPERIMENTAL=1
    ;;
  *)
    echo "Usage: $0 [--all|--supported-only|--experimental-only]" >&2
    exit 2
    ;;
esac

run_supported() {
  local version="$1"
  local env_dir="/tmp/pytides-matrix-${version}"

  echo "=== Supported Python ${version} ==="
  rm -rf "$env_dir"
  uv venv --python "$version" "$env_dir"
  uv pip install --python "$env_dir/bin/python" pytest -e "$PROJECT_ROOT"
  "$env_dir/bin/python" -m pytest -q "$PROJECT_ROOT/tests"
}

run_experimental() {
  local version="$1"
  local env_dir="/tmp/pytides-matrix-exp-${version}"

  echo "=== Experimental Python ${version} ==="
  echo "Running without package install because ${version} is outside project.requires-python."
  rm -rf "$env_dir"
  uv venv --python "$version" "$env_dir"
  uv pip install --python "$env_dir/bin/python" pytest numpy scipy
  PYTHONPATH="$PROJECT_ROOT" "$env_dir/bin/python" -m pytest -q "$PROJECT_ROOT/tests"
}

export UV_CACHE_DIR="$CACHE_DIR"

cd "$PROJECT_ROOT"

if [[ "$RUN_SUPPORTED" -eq 1 ]]; then
  for version in "${SUPPORTED_VERSIONS[@]}"; do
    run_supported "$version"
  done
fi

if [[ "$RUN_EXPERIMENTAL" -eq 1 ]]; then
  for version in "${EXPERIMENTAL_VERSIONS[@]}"; do
    run_experimental "$version"
  done
fi

echo "Python matrix test run completed."
