"""Frozen categorical-local alternative with separate results files."""
from .check_controls import main as controls
from .check_neural import main as neural

if __name__ == '__main__':
    controls(local=True, outname='local_controls_results.json')
    neural(recipes=('residual_local',), outname='local_results.json')
