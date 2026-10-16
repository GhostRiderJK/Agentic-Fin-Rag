"""OCR fallback for scanned PDFs (Tesseract via pytesseract).

Used only when the parser reports ``needs_ocr=True``. Renders each PDF page
to an image with pdf2image and runs Tesseract on it.
"""

from __future__ import annotations

from pathlib import Path

from app.core.logging import get_logger
from app.ingestion.parser import ParsedDocument, ParsedPage, TableExtract

log = get_logger(__name__)


def ocr_pdf(path: str | Path) -> ParsedDocument:
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError as e:  # pragma: no cover
        log.warning("OCR dependencies not installed (%s); skipping OCR", e)
        return _empty()

    pages_text: list[str] = []
    pages: list[ParsedPage] = []
    try:
        images = convert_from_path(str(path), dpi=200)
    except Exception as e:  # pragma: no cover
        log.warning("pdf2image failed (%s); is poppler installed?", e)
        return _empty()

    for i, img in enumerate(images, start=1):
        text = pytesseract.image_to_string(img)
        pages.append(ParsedPage(page=i, text=text))
        pages_text.append(text)

    return ParsedDocument(
        pages=pages,
        raw_text="\n\n".join(pages_text),
        tables=[],
        needs_ocr=False,
        source_format="pdf-ocr",
    )


def _empty() -> ParsedDocument:
    return ParsedDocument(pages=[], raw_text="", tables=[], needs_ocr=False,
                          source_format="pdf-ocr")
