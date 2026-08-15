"""
Rich TUI module for the Research Paper Summarizer.

Provides interactive file selection, progress displays, and formatted output
using the Rich library.
"""

import threading
import time
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

import typer

from gemini_client import ProgressUpdate, read_key_file, save_key_file


# --- First-run API key setup ---


def setup_api_key() -> None:
    """Check for an API key and prompt the user if none is found.

    Called once at startup. If no key is found via env var or existing file,
    prompts the user to paste one and saves it to api_key.txt.
    """
    import os

    # Already set via env var — nothing to do
    if os.environ.get("GOOGLE_API_KEY"):
        return

    # Already have a saved key file — nothing to do
    if read_key_file():
        return

    # First run — prompt the user
    console.print()
    console.print(
        Panel.fit(
            "[bold yellow]🔑 Welcome! Let's set up your API key[/bold yellow]\n\n"
            "This tool needs a [bold]Gemini API key[/bold] to work.\n"
            "Get a free one at: [cyan]https://aistudio.google.com/apikey[/cyan]",
            border_style="yellow",
        )
    )
    console.print()

    while True:
        key = Prompt.ask("[bold]Paste your API key[/bold]").strip()

        if not key:
            console.print("[red]Key cannot be empty. Try again.[/red]")
            continue

        # Basic sanity check — Gemini keys start with "AIza"
        if not key.startswith("AIza"):
            console.print(
                "[yellow]⚠ That doesn't look like a Gemini key (should start with 'AIza').[/yellow]\n"
                "[dim]You can still proceed, but double-check it's correct.[/dim]"
            )

        # Save it
        saved_path = save_key_file(key)
        console.print(f"\n[bold green]✓[/bold green] Key saved to: [white]{saved_path}[/white]")
        console.print("[dim]You won't need to do this again.[/dim]")
        console.print()
        return

console = Console()


# --- PDF file selection ---


def prompt_pdf_path() -> Path:
    """Show an interactive directory browser to select a PDF file.

    Returns:
        A Path object pointing to an existing PDF file.
    """
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]📄 Research Paper Summarizer[/bold cyan]\n"
            "[dim]Powered by Gemini 3.5 Flash Lite[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    current_dir = Path.cwd().resolve()

    while True:
        selected = _browse_directory(current_dir)
        if selected is None:
            # User cancelled
            console.print("[bold red]Cancelled.[/bold red]")
            raise typer.Exit(0)

        if selected.is_dir():
            current_dir = selected
            continue

        if selected.is_file() and selected.suffix.lower() == ".pdf":
            console.print(f"\n[bold green]✓[/bold green] Selected: [white]{selected}[/white]")
            return selected

        if selected.is_file():
            console.print(
                f"\n[bold yellow]⚠[/bold yellow] Not a PDF: [white]{selected.name}[/white]\n"
                "[dim]Please select a file with a .pdf extension.[/dim]"
            )
            # Stay in same directory, let them try again
            continue


def _browse_directory(directory: Path) -> Optional[Path]:
    """Display a directory listing and let the user navigate/select.

    Args:
        directory: The directory to browse.

    Returns:
        The selected Path, or None if cancelled.
    """
    # Gather entries
    try:
        entries = sorted(directory.iterdir(), key=_sort_key)
    except PermissionError:
        console.print(f"[bold red]Permission denied:[/bold red] {directory}")
        return None

    # Display header
    header = Panel(
        f"[bold cyan]📂 {directory}[/bold cyan]",
        border_style="cyan",
    )
    console.print(header)

    # Build table
    table = Table(
        show_header=False,
        box=None,
        padding=(0, 1),
        expand=True,
    )
    table.add_column("", style="bold", width=4)
    table.add_column("", width=2)
    table.add_column("", ratio=1)

    # Up one level
    if directory.parent != directory:
        table.add_row(" 0 ", "📁", "[bold cyan]..[/bold cyan] (up)")

    # List entries
    idx = 1
    for entry in entries:
        if entry.is_dir():
            table.add_row(f" {idx:2d} ", "📁", f"[cyan]{entry.name}/[/cyan]")
        elif entry.suffix.lower() == ".pdf":
            table.add_row(f" {idx:2d} ", "📄", f"[green]{entry.name}[/green]")
        else:
            table.add_row(f" {idx:2d} ", " ", f"[dim]{entry.name}[/dim]")
        idx += 1

    console.print(table)
    console.print("[dim]Enter a number to navigate/select, [b]b[/b] to go back, [q]q[/q] to quit[/dim]")

    # Get user input
    while True:
        choice = Prompt.ask("[bold]Choice[/bold]", default="").strip().lower()

        if choice == "q":
            return None
        if choice in ("b", "0"):
            return directory.parent

        try:
            num = int(choice)
        except ValueError:
            console.print("[red]Invalid input. Enter a number, 'b', or 'q'.[/red]")
            continue

        if num < 0 or num >= idx:
            console.print(f"[red]Number must be between 0 and {idx - 1}.[/red]")
            continue

        if num == 0:
            return directory.parent

        selected = entries[num - 1]
        return selected


def _sort_key(path: Path) -> tuple:
    """Sort key: directories first, then files, case-insensitive alphabetically."""
    return (0 if path.is_dir() else 1, path.name.lower())


# --- Progress display ---


def make_extraction_progress() -> Progress:
    """Create a Progress bar for the PDF extraction phase."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    )


def run_with_live_progress(
    total_pages: int,
    progress_updates: List[ProgressUpdate],
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Display a live-updating status panel during batch processing.

    Args:
        total_pages: Total number of pages in the PDF.
        progress_updates: Shared list appended to by the processing thread.
        stop_event: When set, the loop exits (worker finished or errored).
    """
    layout = Layout()
    layout.split_column(
        Layout(name="status", size=6),
        Layout(name="details"),
    )

    with Live(layout, console=console, refresh_per_second=2, screen=False) as live:
        last_update: Optional[ProgressUpdate] = None
        displayed_indices = set()

        while True:
            # Check for new updates
            for i, update in enumerate(progress_updates):
                if i not in displayed_indices:
                    displayed_indices.add(i)
                    last_update = update

            if last_update is None:
                layout["status"].update(
                    Panel("Waiting to start...", border_style="blue")
                )
                live.refresh()
                time.sleep(0.5)
                if stop_event and stop_event.is_set():
                    break
                continue

            # Build status display
            status_content = _render_status(last_update, total_pages)
            layout["status"].update(status_content)

            # Build details (history of completed batches)
            detail_lines = []
            for i, upd in enumerate(progress_updates):
                if upd.stage == "batch_processing" and "complete" in upd.message.lower():
                    detail_lines.append(f"  [green]✓[/green] {upd.message}")
                elif upd.stage == "batch_processing" and "ERROR" in upd.message:
                    detail_lines.append(f"  [red]✗[/red] {upd.message}")
                elif upd.stage == "waiting" and upd.wait_seconds == 0:
                    detail_lines.append(f"  [blue]→[/blue] {upd.message}")

            if detail_lines:
                detail_text = "\n".join(detail_lines[-8:])
                layout["details"].update(
                    Panel(detail_text, title="[dim]History[/dim]", border_style="dim")
                )
            else:
                layout["details"].update(
                    Panel("[dim]Waiting for batches to complete...[/dim]", border_style="dim")
                )

            # Stop when done update arrives or worker finished/errored
            if last_update.stage == "done":
                break
            if stop_event and stop_event.is_set():
                break

            time.sleep(0.5)


def _render_status(update: ProgressUpdate, total_pages: int) -> Panel:
    """Render a status panel for the current progress update."""
    if update.stage == "extracting":
        return Panel(
            Text.assemble(
                ("📖 Extracting pages...\n", "bold"),
                (f"  Pages: {update.pages_processed}/{update.total_pages}", ""),
            ),
            title="[bold yellow]Extraction[/bold yellow]",
            border_style="yellow",
        )

    elif update.stage == "batch_processing":
        bar = _make_simple_bar(update.pages_processed, total_pages, 20)
        return Panel(
            Text.assemble(
                ("🤖 Summarizing with Gemini\n", "bold"),
                (f"  {bar} {update.pages_processed}/{total_pages} pages\n", ""),
                (f"  Batch {update.batch_number}/{update.total_batches}", "cyan"),
            ),
            title="[bold green]Processing[/bold green]",
            border_style="green",
        )

    elif update.stage == "waiting":
        wait_m = update.wait_seconds // 60
        wait_s = update.wait_seconds % 60
        time_str = f"{wait_m}m {wait_s}s" if wait_m > 0 else f"{wait_s}s"
        return Panel(
            Text.assemble(
                ("⏳ Rate limit cooldown\n", "bold yellow"),
                (f"  Next batch in: {time_str}\n", "yellow"),
                (f"  Batch {update.batch_number}/{update.total_batches} queued", "dim"),
            ),
            title="[bold yellow]Waiting[/bold yellow]",
            border_style="yellow",
        )

    elif update.stage == "synthesizing":
        return Panel(
            Text.assemble(
                ("🧠 Synthesizing final summary...\n", "bold cyan"),
                ("  Combining all observations into a structured breakdown", "dim"),
            ),
            title="[bold cyan]Synthesis[/bold cyan]",
            border_style="cyan",
        )

    elif update.stage == "done":
        return Panel(
            Text.assemble(
                ("✅ Complete!\n", "bold green"),
                ("  Summary ready for display", "dim"),
            ),
            title="[bold green]Done[/bold green]",
            border_style="green",
        )

    return Panel(update.message or "Processing...", border_style="blue")


def _make_simple_bar(value: int, maximum: int, width: int) -> str:
    """Make a simple text-based progress bar."""
    if maximum == 0:
        return "[" + " " * width + "]"
    filled = int(value / maximum * width)
    filled = min(filled, width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"


# --- Output display ---


def display_summary(summary: str, pdf_path: Path, save: bool = True) -> None:
    """Display the final summary in a Rich-formatted layout and optionally save to file.

    Args:
        summary: The markdown-formatted summary string.
        pdf_path: Path to the original PDF (used for naming the output file).
        save: Whether to save the summary to a .md file (default: True).
    """
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]📄 Research Paper Summary[/bold cyan]",
            border_style="cyan",
        )
    )
    console.print()

    # Render markdown
    md = Markdown(summary)
    console.print(md)

    # Save to file
    if save:
        output_path = pdf_path.with_suffix(".summary.md")
        output_path.write_text(summary, encoding="utf-8")
        console.print()
        console.print(
            f"[bold green]✓[/bold green] Summary saved to: [white]{output_path}[/white]"
        )
    console.print()


def show_error(message: str, detail: Optional[str] = None) -> None:
    """Display an error message in a styled panel."""
    content = Text(message, style="bold red")
    if detail:
        content.append(f"\n\n{detail}", style="dim")
    console.print(Panel(content, title="[red]Error[/red]", border_style="red"))


def show_info(message: str) -> None:
    """Display an informational message."""
    console.print(f"[blue]ℹ[/blue] {message}")


def show_success(message: str) -> None:
    """Display a success message."""
    console.print(f"[green]✓[/green] {message}")


# --- Interactive Q&A ---


def qa_loop(
    pages: List,
    observations: List[str],
    answer_fn,
) -> bool:
    """Interactive Q&A loop after the summary is displayed.

    Args:
        pages: List of (page_number, text, image) tuples.
        observations: List of observation strings from batch processing.
        answer_fn: Function that takes a question string and returns an answer string.

    Returns:
        True if the user wants to start a new paper, False to exit.
    """
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]💬 Ask questions about this paper[/bold cyan]\n"
            "[dim]Type your question, [green]:new[/green] for a new paper, or [red]:q[/red] to quit[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    while True:
        raw = Prompt.ask("[bold]Your question[/bold]").strip()

        if raw.lower() in (":q", "quit", "exit"):
            console.print("[dim]Goodbye![/dim]")
            return False

        if raw.lower() == ":new":
            console.print("[dim]Starting a new paper...[/dim]")
            return True

        if not raw:
            continue

        console.print()
        with console.status("[bold cyan]Thinking...[/bold cyan]", spinner="dots"):
            try:
                answer = answer_fn(raw)
            except Exception as e:
                answer = f"[red]Error: {e}[/red]"

        console.print()
        console.print(
            Panel(
                Markdown(answer),
                title="[bold green]Answer[/bold green]",
                border_style="green",
            )
        )
        console.print()
        console.print("[dim]───[/dim]")