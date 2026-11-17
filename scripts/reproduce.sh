#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
environment=scikit-longitudinal-tpt
conda env create -f environment.yml
conda run -n "$environment" bash setup-render-ci.sh
conda run -n "$environment" bash scripts/render_local.sh computo-html
conda run -n "$environment" bash scripts/render_local.sh computo-pdf
