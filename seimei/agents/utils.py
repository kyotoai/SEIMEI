from __future__ import annotations

from pathlib import Path
from typing import Optional


def pdf_bytes_to_text(data: bytes) -> str:
    """Extract text from PDF bytes. Returns an empty string on failure."""
    try:
        import pymupdf  # package: PyMuPDF
    except Exception:
        return ""

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
        parts = [page.get_text() or "" for page in doc]
        doc.close()
        return "\n".join(parts).strip()
    except Exception:
        return ""


def view_pdf_text(path: str, max_chars: Optional[int] = None) -> str:
    """
    Read a local PDF and return extracted text.
    Returns an error string when the file cannot be read or parsed.
    """
    pdf_path = Path(path).expanduser()
    if not pdf_path.exists():
        return f"[pdf_view_error] file not found: {pdf_path}"
    if not pdf_path.is_file():
        return f"[pdf_view_error] not a file: {pdf_path}"

    try:
        data = pdf_path.read_bytes()
    except Exception as exc:
        return f"[pdf_view_error] failed to read file: {exc}"

    text = pdf_bytes_to_text(data)
    if not text:
        return "[pdf_view_error] failed to extract text (PyMuPDF may be missing or the PDF has no extractable text)"
    if max_chars is not None and max_chars > 0:
        return text[:max_chars]
    return text
