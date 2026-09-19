#!/bin/bash
# Download the pretrained products used by tutorial.ipynb into tutorial/data/:
#   data/models/    weights of the best Optuna trial (DeepWiener, SO-like mask, inhomogeneous noise)
#   data/studies/   Optuna study (sqlite) used to pick the best trial
#   data/spectrum/  OQE Fisher matrix and noise bias (2000 simulations each)
#   data/aux/       2D noise variance map, cobaya covariances, NaMaster noise bias
#   data/masks/     SO-like hits map
#
# Usage (from the tutorial/ folder):   bash get_tutorial_data.sh

set -euo pipefail

URL="https://github.com/Belencostanza/CMBtorch/releases/download/tutorial-data-v1/tutorial_data.tar.gz"
SHA256="f8a829aa43f6141586ffd1ca8884f01faffd63086957efd149d30a9e5e7cd83b"

cd "$(dirname "$0")"

if [ -d data/models ] && [ -n "$(ls -A data/models 2>/dev/null)" ]; then
    echo "tutorial/data/ already present, nothing to do."
    exit 0
fi

echo "Downloading $URL"
if command -v curl >/dev/null; then
    curl -L --fail -o tutorial_data.tar.gz "$URL"
else
    wget -O tutorial_data.tar.gz "$URL"
fi

echo "$SHA256  tutorial_data.tar.gz" | sha256sum -c -
tar -xzf tutorial_data.tar.gz
rm tutorial_data.tar.gz
echo "Done: tutorial/data/"
