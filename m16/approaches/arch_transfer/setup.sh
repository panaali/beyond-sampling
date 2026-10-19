#!/bin/bash
# Fresh-container setup for T432 helpers: .venv with CPU torch, numpy, scipy.
set -e
REPO=$(cd "$(dirname "$0")/../../.." && pwd); cd "$REPO"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
pip install -q numpy scipy
git config user.email >/dev/null || git config user.email noreply@anthropic.com
git config user.name >/dev/null || git config user.name Claude
python -c "import torch; print('setup ok', torch.__version__)"
