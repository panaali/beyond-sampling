#!/bin/sh
# Refresh the T421 registry and weights (untracked here; never committed on this branch).
cd "$(dirname "$0")/../../.." && git fetch -q origin claude/t421-scale-study && \
git checkout -q origin/claude/t421-scale-study -- m16/approaches/scale_study && git reset -q m16/approaches/scale_study
