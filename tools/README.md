# tools/: documentation tooling

↑ [Project README](../README.md)

| Script | Use |
|---|---|
| [check_links.py](check_links.py) | `python tools/check_links.py`: verifies every relative link in the docs (exit 1 if any is broken) |
| [linkify.py](linkify.py) | `python tools/linkify.py FILE.md ...`: turns backticked repo paths into relative links (idempotent) |
