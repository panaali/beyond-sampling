"""Same pinned corpus runner/objectives, local additional moment shortcut only."""
from pathlib import Path
from ..positive_mass_budget import run as original
from .primitive import estimate
if __name__=='__main__':
    original.HERE=Path(__file__).parent;original.estimate=estimate;original.main()
