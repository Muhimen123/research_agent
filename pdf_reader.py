"""
PDF extraction module.

Uses pymupdf to extract text and render page images from a PDF.
Returns a list of (page_number, text, image) tuples for downstream processing.
"""

from pathlib import Path
from typing import List, Tuple

import pymupdf
from PIL import Image


def extract_pdf(
    pdf_path: str | Path,
    dpi: int = 200,
) -> List[Tuple[int, str, Image.Image]]:
    """Extract text and render each page of a PDF as a PIL Image.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution for page rendering (higher = sharper images, more tokens).

    Returns:
        A list of (page_number, text, image) tuples, 1-indexed page numbers.

    Raises:
        FileNotFoundError: If the PDF doesn't exist.
        pymupdf.FileDataError: If the PDF is corrupted or invalid.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = pymupdf.open(str(pdf_path))
    pages: List[Tuple[int, str, Image.Image]] = []

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]

            # Extract text
            text = page.get_text()

            # Render page as PIL Image
            zoom = dpi / 72  # Default pymupdf DPI is 72
            mat = pymupdf.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            pages.append((page_num + 1, text, img))

    finally:
        doc.close()

    return pages


def extract_text_only(pdf_path: str | Path) -> List[Tuple[int, str]]:
    """Extract only text from each page (no images). Useful for quick previews.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A list of (page_number, text) tuples.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = pymupdf.open(str(pdf_path))
    pages: List[Tuple[int, str]] = []

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            pages.append((page_num + 1, text))
    finally:
        doc.close()

    return pages