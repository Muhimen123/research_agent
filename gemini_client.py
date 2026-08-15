"""
Gemini integration module.

Handles sending PDF pages (text + images) to Gemini 3.5 Flash Lite in batches
with rate limiting, accumulating observations, and producing a final structured summary.
"""

import io
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from google.genai import Client
from google.genai import types as genai_types
from PIL import Image

# --- Data structures ---

PageData = Tuple[int, str, Image.Image]  # (page_number, text, image)


@dataclass
class ProgressUpdate:
    """Progress information emitted during batch processing."""

    stage: str  # "extracting", "batch_processing", "waiting", "synthesizing", "done"
    pages_processed: int = 0
    total_pages: int = 0
    batch_number: int = 0
    total_batches: int = 0
    message: str = ""
    wait_seconds: int = 0


ProgressCallback = Callable[[ProgressUpdate], None]


# --- Prompts ---

BATCH_PROMPT = """You are analyzing a research paper. Below are pages {page_range} of the paper, including both the extracted text and rendered page images.

For each page, extract:
1. **Key observations** — what important claims, data, or findings are presented?
2. **Figures/tables** — note any figures, charts, or tables and what they show.
3. **Terminology** — any new terms, acronyms, or definitions introduced.

Focus on factual extraction. Do NOT summarize the whole paper yet — just capture what's on these pages.

Format your response as a bullet list per page."""

SYNTHESIS_PROMPT = """You are a research paper analyst. Below are observations extracted page-by-page from a research paper.

Synthesize these observations into a well-structured summary with the following sections:

## Introduction
- What problem does the paper address?
- What are the authors' main motivations?
- What is the key research question or hypothesis?

## Methodology
- What approach/method did the authors use?
- Key architectural or experimental design decisions?
- Any novel techniques introduced?

## Experiments
- What datasets were used?
- What metrics were used for evaluation?
- What baselines or comparison methods were used?

## Results
- What were the main quantitative results?
- Key findings from figures/tables?
- Any surprising or negative results?

## Conclusion
- What do the authors conclude?
- What limitations do they acknowledge?
- What future work do they suggest?

## Key Contributions
- 3-5 bullet points listing the main contributions of this paper

Make the summary comprehensive but concise. Use markdown formatting."""


# --- Core functions ---


def _get_client(api_key: Optional[str] = None) -> Client:
    """Create a Gemini client using the provided API key.

    Key resolution order:
    1. `api_key` argument
    2. `GOOGLE_API_KEY` environment variable
    3. `api_key.txt` file in the project directory
    4. `~/.config/research_agent/api_key.txt`
    5. `~/.gemini/api_key`
    """
    key = api_key or os.environ.get("GOOGLE_API_KEY") or read_key_file()
    if not key:
        raise ValueError(
            "No API key found. "
            "Set GOOGLE_API_KEY environment variable or create an api_key.txt file "
            "in the project directory with your Gemini API key."
        )
    return Client(api_key=key)


def read_key_file() -> Optional[str]:
    """Try to read the API key from a plain text file in known locations."""
    locations = [
        Path("api_key.txt"),  # project root
        Path.home() / ".config" / "research_agent" / "api_key.txt",
        Path.home() / ".gemini" / "api_key",
    ]
    for path in locations:
        try:
            if path.exists():
                for line in path.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        return line
        except OSError:
            continue
    return None


def save_key_file(key: str, path: Optional[Path] = None) -> Path:
    """Save an API key to a file.

    Args:
        key: The API key to save.
        path: Where to save it (default: api_key.txt in project root).

    Returns:
        The path the key was saved to.
    """
    if path is None:
        path = Path("api_key.txt")
    path.write_text(key.strip() + "\n")
    return path


def _call_gemini(
    client: Client,
    model: str,
    parts: list,
    timeout_ms: int = 300000,  # 5 minutes
) -> str:
    """Send a multimodal request to Gemini and return the response text.

    Args:
        client: Gemini client instance.
        model: Model name.
        parts: List of content parts (text + images).
        timeout_ms: Request timeout in milliseconds (default: 300000 = 5 min).

    Returns:
        Response text from the model.
    """
    config = genai_types.GenerateContentConfig(
        http_options=genai_types.HttpOptions(timeout=timeout_ms),
        automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(
            disable=True,
        ),
    )
    response = client.models.generate_content(
        model=model,
        contents=parts,
        config=config,
    )
    return response.text


def _build_batch_parts(
    pages: List[PageData],
    start_idx: int,
    batch_size: int,
) -> Tuple[List, str, int]:
    """Build the list of Part objects and text prompt for a batch of pages."""
    batch = pages[start_idx : start_idx + batch_size]
    first_page = batch[0][0]
    last_page = batch[-1][0]
    page_range = f"{first_page}–{last_page}"

    parts: list = []
    # Start with the prompt text
    prompt = BATCH_PROMPT.format(page_range=page_range)
    parts.append(genai_types.Part.from_text(text=prompt))

    # Add each page's content
    for page_num, text, img in batch:
        # Add page text
        parts.append(
            genai_types.Part.from_text(text=f"\n\n--- Page {page_num} text ---\n{text}")
        )
        # Add page image
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        parts.append(
            genai_types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")
        )

    return parts, page_range, len(batch)


def process_pdf_in_batches(
    pages: List[PageData],
    batch_size: int = 5,
    model: str = "gemini-3.5-flash-lite",
    api_key: Optional[str] = None,
    grace_period: int = 5,
    progress_callback: Optional[ProgressCallback] = None,
) -> List[str]:
    """Process PDF pages in batches with rate limiting.

    Sends pages in small batches to Gemini, with a grace period between each
    batch to avoid rate limits. Accumulates observations from all batches.

    Args:
        pages: List of (page_number, text, image) tuples.
        batch_size: Number of pages per batch (default: 5).
        model: Gemini model name (default: gemini-3.5-flash-lite).
        api_key: Optional API key (defaults to GOOGLE_API_KEY env var).
        grace_period: Seconds to wait between batches (default: 5).
        progress_callback: Optional function to receive progress updates.

    Returns:
        A list of observation strings, one per batch.
    """
    client = _get_client(api_key)
    total_pages = len(pages)
    total_batches = (total_pages + batch_size - 1) // batch_size
    observations: List[str] = []

    if progress_callback:
        progress_callback(
            ProgressUpdate(
                stage="batch_processing",
                pages_processed=0,
                total_pages=total_pages,
                batch_number=0,
                total_batches=total_batches,
                message=f"Starting batch processing of {total_pages} pages in {total_batches} batches",
            )
        )

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        batch_number = batch_idx + 1

        # --- Wait before this batch (except the first one) ---
        if batch_idx > 0:
            _do_countdown_wait(grace_period, batch_number, total_batches, progress_callback)

        # --- Build and send this batch ---
        if progress_callback:
            progress_callback(
                ProgressUpdate(
                    stage="batch_processing",
                    pages_processed=start_idx,
                    total_pages=total_pages,
                    batch_number=batch_number,
                    total_batches=total_batches,
                    message=f"Sending batch {batch_number}/{total_batches} (pages {start_idx + 1}–{min(start_idx + batch_size, total_pages)})...",
                )
            )

        parts, page_range, batch_len = _build_batch_parts(
            pages, start_idx, batch_size
        )

        try:
            response_text = _call_gemini(client, model, parts)
            observations.append(response_text)

            pages_done = min(start_idx + batch_size, total_pages)
            if progress_callback:
                progress_callback(
                    ProgressUpdate(
                        stage="batch_processing",
                        pages_processed=pages_done,
                        total_pages=total_pages,
                        batch_number=batch_number,
                        total_batches=total_batches,
                        message=f"Batch {batch_number}/{total_batches} complete — processed pages 1–{pages_done}",
                    )
                )
        except Exception as e:
            error_msg = f"[Error processing batch {batch_number} (pages {page_range}): {e}]"
            observations.append(error_msg)
            if progress_callback:
                progress_callback(
                    ProgressUpdate(
                        stage="batch_processing",
                        pages_processed=start_idx,
                        total_pages=total_pages,
                        batch_number=batch_number,
                        total_batches=total_batches,
                        message=f"ERROR on batch {batch_number}: {e}",
                    )
                )

    if progress_callback:
        progress_callback(
            ProgressUpdate(
                stage="done",
                pages_processed=total_pages,
                total_pages=total_pages,
                batch_number=total_batches,
                total_batches=total_batches,
                message=f"All {total_batches} batches processed — {len(observations)} observations collected",
            )
        )

    return observations


def synthesize_summary(
    observations: List[str],
    model: str = "gemini-3.5-flash-lite",
    api_key: Optional[str] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> str:
    """Synthesize all page observations into a final structured summary.

    Args:
        observations: List of observation strings from each batch.
        model: Gemini model name.
        api_key: Optional API key (defaults to GOOGLE_API_KEY env var).
        progress_callback: Optional function to receive progress updates.

    Returns:
        The final structured summary as a markdown string.
    """
    client = _get_client(api_key)

    if progress_callback:
        progress_callback(
            ProgressUpdate(
                stage="synthesizing",
                message="Synthesizing final summary from all observations...",
            )
        )

    # Combine all observations
    combined = "\n\n".join(
        f"## Observations from batch {i+1}\n{obs}"
        for i, obs in enumerate(observations)
    )

    parts = [
        genai_types.Part.from_text(text=SYNTHESIS_PROMPT),
        genai_types.Part.from_text(text=f"\n\n--- Page-by-page observations ---\n\n{combined}"),
    ]

    try:
        summary = _call_gemini(client, model, parts)
    except Exception as e:
        summary = f"Error generating summary: {e}"

    if progress_callback:
        progress_callback(
            ProgressUpdate(
                stage="done",
                message="Summary complete!",
            )
        )

    return summary


QA_PROMPT = """You are an AI research assistant analyzing a specific paper. Below is the full extracted content of the paper (text from each page) along with observations from a page-by-page analysis.

Use this content to answer the user's question. Be precise, cite specific details from the paper, and if the paper doesn't contain the information needed, say so clearly.

Paper content:
{paper_context}

---

User question: {question}

Answer the question based only on the paper content above."""


def answer_question(
    question: str,
    pages: List[PageData],
    observations: List[str],
    model: str = "gemini-3.5-flash-lite",
    api_key: Optional[str] = None,
) -> str:
    """Answer a question about the paper using Gemini.

    Args:
        question: The user's question.
        pages: List of (page_number, text, image) tuples.
        observations: List of observation strings from batch processing.
        model: Gemini model name.
        api_key: Optional API key (defaults to GOOGLE_API_KEY env var).

    Returns:
        The answer text.
    """
    client = _get_client(api_key)

    # Build context from all pages
    text_context = "\n\n".join(
        f"--- Page {num} ---\n{text}" for num, text, _ in pages
    )
    observations_context = "\n\n".join(
        f"## Batch {i+1} observations\n{obs}"
        for i, obs in enumerate(observations)
    )
    full_context = (
        f"--- Full paper text ---\n\n{text_context}\n\n"
        f"--- Page-by-page observations ---\n\n{observations_context}"
    )

    prompt = QA_PROMPT.format(paper_context=full_context, question=question)

    # Include page images for visual questions
    parts = [genai_types.Part.from_text(text=prompt)]
    for _, _, img in pages:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        parts.append(genai_types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"))

    try:
        response = client.models.generate_content(
            model=model,
            contents=parts,
            config=genai_types.GenerateContentConfig(
                http_options=genai_types.HttpOptions(timeout=300000),
                automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(
                    disable=True,
                ),
            ),
        )
        return response.text
    except Exception as e:
        return f"Error answering question: {e}"


# --- Helpers ---


def _do_countdown_wait(
    seconds: int,
    next_batch: int,
    total_batches: int,
    progress_callback: Optional[ProgressCallback] = None,
):
    """Wait for `seconds` while reporting countdown progress."""
    if progress_callback:
        progress_callback(
            ProgressUpdate(
                stage="waiting",
                wait_seconds=seconds,
                batch_number=next_batch,
                total_batches=total_batches,
                message=f"Waiting {seconds}s before batch {next_batch}/{total_batches}...",
            )
        )
    # Sleep in 1-second increments so the user can see progress
    # For long waits, report every 60s to reduce callback noise
    report_interval = 60 if seconds > 120 else 10
    remaining = seconds
    while remaining > 0:
        chunk = min(report_interval, remaining)
        time.sleep(chunk)
        remaining -= chunk
        if progress_callback and remaining > 0:
            progress_callback(
                ProgressUpdate(
                    stage="waiting",
                    wait_seconds=remaining,
                    batch_number=next_batch,
                    total_batches=total_batches,
                    message=f"Waiting {remaining}s before batch {next_batch}/{total_batches}...",
                )
            )
    if progress_callback:
        progress_callback(
            ProgressUpdate(
                stage="waiting",
                wait_seconds=0,
                batch_number=next_batch,
                total_batches=total_batches,
                message=f"Wait over — starting batch {next_batch}/{total_batches}",
            )
        )