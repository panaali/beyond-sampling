"""Turn backticked repo paths in markdown docs into relative links (only paths that exist; idempotent).

usage: python tools/linkify.py FILE.md [FILE.md ...]
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAT = re.compile(r"(?<!\[)`([A-Za-z0-9_./\-*]+?)`(?!\]\()")


def resolve(token, doc_dir):
    t = token.rstrip("/")
    if "*" in t:
        t = t.split("*")[0].rstrip("/")
    for base in (doc_dir, ROOT, os.path.join(ROOT, "m16")):
        cand = os.path.normpath(os.path.join(base, t))
        if t and os.path.exists(cand) and cand.startswith(ROOT) and "/.venv" not in cand and "/data/" not in cand + "/":
            return os.path.relpath(cand, doc_dir)
    return None


def linkify(path):
    doc_dir = os.path.dirname(os.path.abspath(path))
    s = open(path).read()
    out, n = [], 0
    in_code = False
    for line in s.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
        if not in_code:
            def sub(m):
                nonlocal n
                rel = resolve(m.group(1), doc_dir)
                if rel is None:
                    return m.group(0)
                n += 1
                return f"[`{m.group(1)}`]({rel})"
            line = PAT.sub(sub, line)
        out.append(line)
    open(path, "w").write("\n".join(out))
    return n


if __name__ == "__main__":
    for f in sys.argv[1:]:
        print(f, linkify(f))
