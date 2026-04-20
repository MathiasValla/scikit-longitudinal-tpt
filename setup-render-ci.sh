#!/usr/bin/env bash

set -euo pipefail

# scikit-longitudinal installs plain scikit-learn as a dependency, while the
# paper examples need the extended sklearn tree backend shipped by
# scikit-lexicographical-trees. Reinstall it last so its sklearn modules are
# the ones imported during Quarto rendering.
python -m pip install --no-deps --force-reinstall scikit-lexicographical-trees==0.0.5
