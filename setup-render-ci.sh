#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
wheel_dir="$(mktemp -d)"
trap 'rm -rf "$wheel_dir"' EXIT
python scripts/build_tpt_backend.py --wheel-dir "$wheel_dir"
wheel=("$wheel_dir"/scikit_lexicographical_trees-*.whl)

python -m pip install "${wheel[0]}" \
  "git+https://github.com/simonprovost/scikit-longitudinal.git@3f20b3d778446deb53c24fe6dcc2eae4c9d6d450"

# Plain scikit-learn is a transitive dependency and shares the sklearn
# namespace. Reinstall the article's backend last to ensure it is imported.
python -m pip install --force-reinstall --no-deps "${wheel[0]}"
