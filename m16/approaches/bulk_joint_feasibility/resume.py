"""Finish only the unchanged predeclared panel after independent phase1 audit."""
from pathlib import Path
import panel
P=Path(__file__).resolve().parent
panel.main(P/'panel.json','panel_phase2.json')
