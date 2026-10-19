"""Build the tutorial page from tutorial/src/page.html and tutorial/data/*.json (T374).

Writes two files:
  tutorial/index.html     a complete standalone HTML document (open it in any browser)
  tutorial/artifact.html  the same page without the <!doctype>/<html>/<body> wrapper, for publishing as an Artifact
Run after editing tutorial/data/status.json:  python3 tutorial/tools/build.py
"""
import json, os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def load(name):
    with open(os.path.join(ROOT, "data", name)) as f:
        return json.dumps(json.load(f), separators=(",", ":")).replace("</", "<\\/")


def main():
    page = open(os.path.join(ROOT, "src", "page.html")).read()
    for key, name in [("__WEIGHTS__", "weights.json"), ("__PANELS__", "panels.json"), ("__STATUS__", "status.json"), ("__FINDINGS__", "findings.json")]:
        page = page.replace(key, load(name))
    open(os.path.join(ROOT, "artifact.html"), "w").write(page)
    head, _, rest = page.partition("<article>")
    doc = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
           "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
           + head + "</head>\n<body>\n<article>" + rest + "\n</body>\n</html>\n")
    open(os.path.join(ROOT, "index.html"), "w").write(doc)
    print("wrote index.html", len(doc), "bytes; artifact.html", len(page), "bytes")


if __name__ == "__main__":
    main()
