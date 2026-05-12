from pathlib import Path
from typing import Any

import pypdf


def load_pdf(path: str) -> list[dict[str, Any]]:
    """
    Extract text from a PDF file page by page.

    Args:
        path: absolute or relative path to the PDF file

    Returns:
        list of dicts with keys: text, source, page
    """
    reader = pypdf.PdfReader(path)
    source = Path(path).name
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        pages.append(
            {
                "text": text,
                "source": source,
                "page": page_number,
            }
        )

    return pages


def load_markdown(path: str) -> list[dict[str, Any]]:
    """
    Load a Markdown file as a single document.

    Args:
        path: absolute or relative path to the .md file

    Returns:
        list with a single dict with keys: text, source, page
    """
    source = Path(path).name
    text = Path(path).read_text(encoding="utf-8")

    return [{"text": text, "source": source, "page": 1}]
