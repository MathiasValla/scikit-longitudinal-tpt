#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUARTO_ENV="${QUARTO_ENV:-}"
QUARTO_BIN="${QUARTO_BIN:-}"
PYTHON_BIN="${QUARTO_PYTHON:-}"
TEXLIVE_BIN="${QUARTO_TEXLIVE_BINPATH:-$HOME/Library/TinyTeX/bin/universal-darwin}"
FORMAT="${1:-computo-html}"

if [[ -z "$QUARTO_BIN" && -n "$QUARTO_ENV" ]]; then
  QUARTO_BIN="$QUARTO_ENV/bin/quarto"
fi

if [[ -z "$QUARTO_BIN" ]]; then
  QUARTO_BIN="$(command -v quarto || true)"
fi

if [[ ! -x "$QUARTO_BIN" ]]; then
  echo "Quarto binary not found. Set QUARTO_BIN or QUARTO_ENV." >&2
  exit 1
fi

if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python || command -v python3 || true)"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python binary not found. Set QUARTO_PYTHON if needed." >&2
  exit 1
fi

if [[ -d "$TEXLIVE_BIN" ]]; then
  export PATH="$TEXLIVE_BIN:$PATH"
fi

export LANG="${QUARTO_LANG:-en_US.UTF-8}"
export LC_ALL="${QUARTO_LC_ALL:-en_US.UTF-8}"

if [[ -n "$QUARTO_ENV" ]]; then
  export QUARTO_SHARE_PATH="${QUARTO_SHARE_PATH:-$QUARTO_ENV/share/quarto}"
  export QUARTO_DENO="${QUARTO_DENO:-$QUARTO_ENV/bin/deno}"
  export DENO_DOM_PLUGIN="${DENO_DOM_PLUGIN:-$QUARTO_ENV/lib/deno_dom.dylib}"
  export QUARTO_DART_SASS="${QUARTO_DART_SASS:-$QUARTO_ENV/bin/sass}"
  export QUARTO_ESBUILD="${QUARTO_ESBUILD:-$QUARTO_ENV/bin/esbuild}"
  export QUARTO_TYPST="${QUARTO_TYPST:-$QUARTO_ENV/bin/typst}"
  export QUARTO_PANDOC="${QUARTO_PANDOC:-$QUARTO_ENV/bin/pandoc}"
fi

export QUARTO_PYTHON="$PYTHON_BIN"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$ROOT"

cd "$ROOT"
exec "$QUARTO_BIN" render paper.qmd --to "$FORMAT"
