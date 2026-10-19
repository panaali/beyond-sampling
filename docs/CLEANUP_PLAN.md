# T396 docs cleanup: plan (before/after map)

↑ [Project README](../README.md) · [TASKS](../TASKS.md) · [AGENTS](../AGENTS.md)

Owner request (10-19 06:5x, verbatim in [OWNER_QA.md](../OWNER_QA.md#directives)): one place per kind of information,
links everywhere else. Rule for this cleanup: **nothing is deleted**. Every moved block lands verbatim in its new home
and leaves a one-line pointer behind. No number or claim status is changed. Research folders and code are not touched.

## One home per kind of information

| Kind of information | Single home | Everywhere else |
|---|---|---|
| Owner questions, answers, directives (verbatim) | [OWNER_QA.md](../OWNER_QA.md) | one-line link |
| Rules that bind agents | [AGENTS.md](../AGENTS.md) (rules + new "Standing rules" lines, each linking to its OWNER_QA quote) | link |
| Protocols (DRI directives 1-8, ladder protocol, continuous-work rule) | new [docs/PROTOCOLS.md](PROTOCOLS.md) | link from TASKS / AGENTS |
| Result numbers, claim status | [INDEX.md](../INDEX.md) (registry + ledger) | claim + link |
| What is running / backlog | [TASKS.md](../TASKS.md) | link |
| Current state in plain words | [STATUS.md](../STATUS.md) (one page) | link |
| DRI process and lessons | [DRI_HANDBOOK.md](../DRI_HANDBOOK.md) | link |
| History (old status updates, finished/superseded board text) | [docs/archive/](archive/) | link |

## Moves, file by file

- **TASKS.md**: owner-directive paragraphs (Fable, no ARC contact, CPU free, parallel sessions) → removed as copies
  (verbatim already in OWNER_QA section 8), replaced by one pointer line. "Continuous work", "Ladder protocol", "DRI
  directives 1-8", "Codex manager update" → docs/PROTOCOLS.md verbatim. "Closed or superseded backlog rows",
  "Recently finished", "Superseded priority lists" → docs/archive/TASKS_history.md verbatim. Kept: claim rules, In
  progress table, Backlog (rows untouched).
- **OWNER_QA.md**: the addendum (6.3-6.5, 4.6-4.8) is filed under its topic sections; the 17:27 Fable directive moves
  into the chronological directive list (section 8, which becomes the chronological index). Every "Later updates"
  paragraph moves verbatim to docs/archive/OWNER_QA_later_updates.md and is replaced by a line naming the INDEX rows
  (T-/E-/H- IDs) it cites plus a link to the archived text.
- **STATUS.md**: rewritten as one screen of current state with INDEX links; the old page (all dated updates and the
  technical summary) → docs/archive/STATUS_history.md verbatim.
- **DRI_HANDBOOK.md**: the six copied "Owner directive" paragraphs at the end → links to OWNER_QA / AGENTS standing
  rules. The 00:1x "batched tutorial update" rule has no verbatim owner quote and is DRI process, so it stays (marked
  paused by 06:4x).
- **AGENTS.md**: add "Standing rules" (one line each, linked to OWNER_QA) and "Where things live" table.
- **README.md**: "Start here" table made identical to AGENTS "Where things live".
- **INDEX.md**: formatting / broken links / exact duplicate rows only.
- **FINDINGS, SCALING_OBSTRUCTIONS, RESULTS, PREREGISTRATION, approaches/README**: checked for long repeats of INDEX
  numbers; anything unclear is listed, not cut.

## After (target line counts, approximate)

TASKS ~250 (from 881), STATUS ~40 (from 377), OWNER_QA ~800 (from 1136), DRI_HANDBOOK ~315; new PROTOCOLS ~70,
archive files hold the rest.
