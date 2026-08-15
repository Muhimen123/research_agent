#!/usr/bin/env python3
"""
Research Paper Summarizer

A TUI tool that extracts text and images from a research paper PDF,
sends them to Gemini 3.5 Flash Lite in batches, and produces a
structured summary (Introduction, Methodology, Experiments, Results,
Conclusion, Key Contributions). After the summary, an interactive Q&A
mode lets you ask follow-up questions about the paper.
"""

import threading
from pathlib import Path
from typing import List, Optional

import typer

from gemini_client import (
    PageData,
    ProgressUpdate,
    answer_question,
    process_pdf_in_batches,
    synthesize_summary,
)
from pdf_reader import extract_pdf
from tui import (
    display_summary,
    make_extraction_progress,
    prompt_pdf_path,
    qa_loop,
    run_with_live_progress,
    setup_api_key,
    show_error,
    show_info,
    show_success,
)

app = typer.Typer(add_completion=False)


def run_pipeline(
    pdf_path: Optional[Path],
    model: str,
    batch_size: int,
    grace_period: int,
) -> None:
    """Core summarization pipeline — shared by CLI and interactive modes."""

    # --- 0. First-run API key setup ---
    setup_api_key()

    # --- 1. Get PDF path ---
    if pdf_path:
        path = pdf_path
        if not path.exists():
            show_error(f"File not found: {path}")
            raise typer.Exit(1)
        if path.suffix.lower() != ".pdf":
            show_error(f"Not a PDF: {path}")
            raise typer.Exit(1)
    else:
        path = prompt_pdf_path()

    summary_path = path.with_suffix(".summary.md")

    # --- 2a. Resume mode: existing summary found ---
    if summary_path.exists():
        show_info(f"Existing summary found: [white]{summary_path.name}[/white]")
        existing_summary = summary_path.read_text(encoding="utf-8")

        # Display the saved summary
        display_summary(existing_summary, path, save=False)

        # Silently extract pages for Q&A context (no progress bars)
        show_info("Loading paper content for Q&A...")
        try:
            pages = extract_pdf(str(path), dpi=200)
            show_success(f"Loaded {len(pages)} page(s)")
        except Exception as e:
            show_error(f"Could not load paper content: {e}")
            raise typer.Exit(1)

        # Enter Q&A directly — no observations since we skipped batch processing
        wants_new = qa_loop(
            pages=pages,
            observations=[],
            answer_fn=lambda q: answer_question(
                question=q,
                pages=pages,
                observations=[],
                model=model,
            ),
        )
        if wants_new:
            run_pipeline(None, model, batch_size, grace_period)
        return

    # --- 2b. Full pipeline: no existing summary ---
    show_info(f"Processing: [white]{path.name}[/white]")
    show_info(f"Model: [cyan]{model}[/cyan] | Batch size: [cyan]{batch_size}[/cyan]")

    show_info("Extracting text and images from PDF...")
    extraction_progress = make_extraction_progress()
    pages: List[PageData] = []

    with extraction_progress:
        task = extraction_progress.add_task("[cyan]Extracting pages...", total=None)
        try:
            raw_pages = extract_pdf(str(path), dpi=200)
            pages = raw_pages
            extraction_progress.update(task, total=len(pages), completed=len(pages))
            extraction_progress.remove_task(task)
        except Exception as e:
            show_error(f"Failed to extract PDF: {e}")
            raise typer.Exit(1)

    show_success(f"Extracted {len(pages)} page(s) with text and images")

    if not pages:
        show_error("No pages found in the PDF.")
        raise typer.Exit(1)

    # --- 3. Process in batches with live progress ---
    show_info("Starting batch summarization...")

    progress_updates: List[ProgressUpdate] = []
    lock = threading.Lock()
    done_event = threading.Event()

    def progress_callback(update: ProgressUpdate):
        with lock:
            progress_updates.append(update)

    progress_callback(
        ProgressUpdate(
            stage="extracting",
            pages_processed=len(pages),
            total_pages=len(pages),
            message=f"Extracted {len(pages)} pages",
        )
    )

    observations: List[str] = []
    synthesis_result: str = ""
    worker_error: Optional[Exception] = None

    def worker():
        nonlocal observations, synthesis_result, worker_error
        try:
            observations = process_pdf_in_batches(
                pages=pages,
                batch_size=batch_size,
                model=model,
                grace_period=grace_period,
                progress_callback=progress_callback,
            )
            synthesis_result = synthesize_summary(
                observations=observations,
                model=model,
                progress_callback=progress_callback,
            )
        except Exception as e:
            worker_error = e
        finally:
            done_event.set()

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    run_with_live_progress(len(pages), progress_updates, stop_event=done_event)

    worker_thread.join()

    if worker_error:
        show_error("An error occurred during processing", detail=str(worker_error))
        raise typer.Exit(1)

    # --- 4. Display and save summary ---
    if synthesis_result:
        display_summary(synthesis_result, path, save=True)
    else:
        show_error("No summary was generated.")
        raise typer.Exit(1)

    # --- 5. Interactive Q&A ---
    wants_new = qa_loop(
        pages=pages,
        observations=observations,
        answer_fn=lambda q: answer_question(
            question=q,
            pages=pages,
            observations=observations,
            model=model,
        ),
    )
    if wants_new:
        # Restart from scratch (new paper selection)
        run_pipeline(None, model, batch_size, grace_period)


@app.command()
def summarize(
    pdf_path: Optional[str] = typer.Argument(
        None, help="Path to the PDF file (omit for interactive TUI)"
    ),
    model: str = typer.Option(
        "gemini-3.5-flash-lite",
        "--model",
        "-m",
        help="Gemini model to use",
    ),
    batch_size: int = typer.Option(
        5,
        "--batch-size",
        "-b",
        help="Number of pages per batch",
        min=1,
        max=10,
    ),
    grace_period: int = typer.Option(
        5,
        "--grace-period",
        "-g",
        help="Seconds to wait between batches (default: 5)",
        min=0,
    ),
):
    """Summarize a research paper PDF using Gemini AI."""
    path = Path(pdf_path).expanduser().resolve() if pdf_path else None
    run_pipeline(path, model, batch_size, grace_period)


@app.command()
def interactive():
    """Launch the interactive TUI to select and summarize a PDF."""
    run_pipeline(None, "gemini-3.5-flash-lite", 5, 5)


if __name__ == "__main__":
    app()