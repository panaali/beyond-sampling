"""Check that every relative markdown link in the repo's docs points to an existing file/dir.

usage: python tools/check_links.py   (exit code 1 if any link is broken)
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINK = re.compile(r"\]\(([^)\s]+)\)")
bad, total = [], 0
for dp, dns, fns in os.walk(ROOT):
    dns[:] = [d for d in dns if d not in (".git", ".venv", "data", "__pycache__")]
    if os.path.relpath(dp, ROOT).startswith(("study/explain", "study/judge")):
        continue          # agent data packets, not documentation
    for f in fns:
        if not f.endswith(".md"):
            continue
        p = os.path.join(dp, f)
        for target in LINK.findall(open(p).read()):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            if "/" not in target and "." not in target:
                continue      # math like [f_1..f_16](x), not a link
            total += 1
            t = target.split("#")[0]
            if t and not os.path.exists(os.path.normpath(os.path.join(dp, t))):
                bad.append((os.path.relpath(p, ROOT), target))
print(f"{total} relative links checked, {len(bad)} broken")
for p, t in bad:
    print(f"  BROKEN {p} -> {t}")
sys.exit(1 if bad else 0)
