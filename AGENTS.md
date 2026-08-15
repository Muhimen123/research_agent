# AGENTS.md — Research Paper Summarizer

## Agent Instructions

This project uses the **task-driven-development** skill for structured, interrupt-safe work.

### Workflow

1. **Plan first, code second** — Before any code changes, read `TASKS.md` to understand current state.
2. **One task at a time** — Only one task is marked `<-- IN PROGRESS` at any point. Its `Notes:` and `Next step:` lines are the single source of truth for where to resume.
3. **Update `TASKS.md` continuously** — Every meaningful checkpoint updates the file. Don't wait until the end.
4. **Re-approval gate** — If the plan materially changes, pause and present the updated plan to the user before continuing.

### Project summary

A Python CLI/TUI tool that:
- Takes a research paper (PDF) as input via a Rich-powered TUI
- Extracts text and renders page images using `pymupdf`
- Sends pages in small batches to **Gemini 3.5 Flash Lite** with a 15-minute grace period between batches
- Produces a full structured breakdown: Introduction, Methodology, Experiments, Results, Conclusion, Key Contributions
- Displays the summary with Rich formatting in the terminal
- Saves the summary as a `.md` file alongside the original PDF

### Key files

- `main.py` — Entry point
- `TASKS.md` — Current task list and progress