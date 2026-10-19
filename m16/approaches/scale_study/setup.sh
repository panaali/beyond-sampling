#!/bin/bash
# Fresh-container setup for T421 helpers: .venv with CPU torch, numpy, blobfile and alg_zoo (editable clone).
set -e
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
[ -d /home/user/alignment-research-center/alg-zoo ] || git clone -q https://github.com/alignment-research-center/alg-zoo /home/user/alignment-research-center/alg-zoo
cd "$REPO"; [ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
pip install -q numpy blobfile
pip install -q -e /home/user/alignment-research-center/alg-zoo
git config user.email >/dev/null || git config user.email noreply@anthropic.com
git config user.name >/dev/null || git config user.name Claude
python -c "import torch, alg_zoo.training; print('setup ok', torch.__version__)"
