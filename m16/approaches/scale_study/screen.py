"""Pre-registered screen (T421 README): seeds >= 2 of a point are skipped when cpu seeds 0 and 1 are both done and both
have accuracy < .80. The anchor (16, 10) is never screened. Exit status 1 = skip."""
import json
import os
import sys

H, T, S = map(int, sys.argv[1:4])
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "models")
if S >= 2 and (H, T) != (16, 10):
    accs = [json.load(open(p))["acc"] for s in (0, 1) if os.path.exists(p := os.path.join(R, f"m_{H}_{T}_cpu{s}.json"))]
    if len(accs) == 2 and max(accs) < 0.80:
        sys.exit(1)
