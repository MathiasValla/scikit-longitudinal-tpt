#!/usr/bin/env bash

set -euo pipefail

# scikit-longitudinal installs plain scikit-learn as a dependency, while the
# paper examples need the extended sklearn tree backend shipped by
# scikit-lexicographical-trees. Reinstall it last so its sklearn modules are
# the ones imported during Quarto rendering. The case study uses a TpT-only
# correction for duration routing and statistics on the pinned release.
cd "$(dirname "${BASH_SOURCE[0]}")"
python scripts/build_tpt_backend.py
