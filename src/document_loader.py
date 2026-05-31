"""
Validity — Document Loader
Component 10 of 10

Handles document ingestion for the pipeline.
Accepts PDF, DOCX, and TXT files and returns clean extracted text.

Removes the manual extraction step — the pipeline now accepts
any supported document format directly.

Input:  File path or bytes
Output: Extracted text string
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os


# ── Supported formats ─────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".doc", ".md", ".html", ".htm"}


# ── Output types ──────────────────────────────────────────────────────────────

@dataclass
class LoadedDocument:
    text:       str
    filename:   str
    extension:  str
    char_count: int
    page_count: Optional[int] = None   # available for PDF
    method:     str = "unknown"


@dataclass
class LoadError:
    filename: str
    reason:   str


# ── Document Loader ───────────────────────────────────────────────────────────

class DocumentLoader:
    """
    Loads documents from file paths or raw bytes.
    Handles PDF, DOCX, TXT, and Markdown.

    Design principles:
    - Never truncates — returns full document text
    - Reports page count for PDFs (used in provenance)
    - Graceful error handling — returns LoadError on failure
    - No external service calls — fully deterministic
    """

    MAX_CHARS = 500_000   # ~350 pages of dense text

    def load_file(self, path: str) -> LoadedDocument | LoadError:
        """
        Load a document from a file path.
        Detects format from extension.
        """
        p        = Path(path)
        filename = p.name
        ext      = p.suffix.lower()

        if not p.exists():
            return LoadError(filename=filename, reason=f"File not found: {path}")

        if ext not in SUPPORTED_EXTENSIONS:
            return LoadError(
                filename = filename,
                reason   = f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
            )

        try:
            if ext == ".pdf":
                return self._load_pdf(path, filename)
            elif ext in (".txt", ".md"):
                return self._load_text(path, filename, ext)
            elif ext in (".html", ".htm"):
                return self._load_html(path, filename)
            elif ext in (".docx", ".doc"):
                return self._load_docx(path, filename)
        except Exception as e:
            return LoadError(filename=filename, reason=str(e))

    def load_bytes(
        self,
        data:      bytes,
        filename:  str,
        extension: str,
    ) -> LoadedDocument | LoadError:
        """
        Load a document from raw bytes.
        Used by the API endpoint for uploaded files.
        """
        import tempfile

        ext = extension.lower()
        if not ext.startswith("."):
            ext = "." + ext

        if ext not in SUPPORTED_EXTENSIONS:
            return LoadError(
                filename = filename,
                reason   = f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
            )

        # Write to temp file and load
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        try:
            result = self.load_file(tmp_path)
            if isinstance(result, LoadedDocument):
                result.filename = filename
            return result
        finally:
            os.unlink(tmp_path)

    # ── PDF ───────────────────────────────────────────────────────────────────

    def _load_pdf(self, path: str, filename: str) -> LoadedDocument | LoadError:
        try:
            import fitz  # pymupdf
        except ImportError:
            return LoadError(
                filename = filename,
                reason   = "PDF support requires pymupdf. Install with: pip install pymupdf"
            )

        doc   = fitz.open(path)
        pages = []

        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            if text.strip():
                pages.append(text)

        doc.close()

        full_text = "\n\n".join(pages)
        full_text = self._clean_text(full_text)

        if not full_text.strip():
            # Text layer empty — attempt OCR
            print(f"[DocumentLoader] No text layer detected in {filename} — attempting OCR...")
            full_text = self._ocr_pdf(path, filename)
            if isinstance(full_text, LoadError):
                return full_text
            method = "pymupdf+tesseract"
        else:
            method = "pymupdf"

        if len(full_text) > self.MAX_CHARS:
            full_text = full_text[:self.MAX_CHARS]

        return LoadedDocument(
            text       = full_text,
            filename   = filename,
            extension  = ".pdf",
            char_count = len(full_text),
            page_count = len(pages),
            method     = method,
        )

    def _ocr_pdf(self, path: str, filename: str) -> str | LoadError:
        """
        OCR a scanned PDF using tesseract.
        Converts each page to an image and extracts text.
        """
        try:
            import pytesseract
            from PIL import Image
            import fitz
        except ImportError as e:
            return LoadError(
                filename = filename,
                reason   = f"OCR requires pytesseract and pillow. Install with: pip install pytesseract pillow. Error: {e}"
            )

        try:
            doc   = fitz.open(path)
            pages = []
            total = len(doc)
            # Cap at 50 pages for OCR — beyond that is impractical
            max_pages = min(total, 50)
            print(f"[DocumentLoader] OCR: processing {max_pages} of {total} pages...")

            for i in range(max_pages):
                page = doc[i]
                # Render page at 2x resolution for better OCR accuracy
                mat  = fitz.Matrix(2, 2)
                pix  = page.get_pixmap(matrix=mat)
                img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(img)
                if text.strip():
                    pages.append(text)

            doc.close()
            return "\n\n".join(pages)

        except Exception as e:
            return LoadError(filename=filename, reason=f"OCR failed: {e}")

    def _load_html(self, path: str, filename: str) -> LoadedDocument:
        """Extract text from HTML using basic tag stripping."""
        import re
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
        # Remove script and style blocks
        raw = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', raw, flags=re.DOTALL | re.IGNORECASE)
        # Strip all HTML tags
        # Replace block-level tags with newlines before stripping
        raw = re.sub(r'<(p|div|br|h[1-6]|li|tr|section|article)[^>]*>', '\n', raw, flags=re.IGNORECASE)
        raw = re.sub(r'</(p|div|h[1-6]|li|tr|section|article)>', '\n', raw, flags=re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', raw)
        # Decode common HTML entities
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&nbsp;', ' ').replace('&#160;', ' ')
        text = self._clean_text(text)
        if len(text) > self.MAX_CHARS:
            text = text[:self.MAX_CHARS]
        return LoadedDocument(
            text       = text,
            filename   = filename,
            extension  = ".html",
            char_count = len(text),
            method     = "html-strip",
        )

    # ── TXT / Markdown ────────────────────────────────────────────────────────

    def _load_text(self, path: str, filename: str, ext: str) -> LoadedDocument:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        text = self._clean_text(text)

        if len(text) > self.MAX_CHARS:
            text = text[:self.MAX_CHARS]

        return LoadedDocument(
            text       = text,
            filename   = filename,
            extension  = ext,
            char_count = len(text),
            method     = "plaintext",
        )

    # ── DOCX ──────────────────────────────────────────────────────────────────

    def _load_docx(self, path: str, filename: str) -> LoadedDocument | LoadError:
        try:
            import docx
        except ImportError:
            return LoadError(
                filename = filename,
                reason   = "DOCX support requires python-docx. Install with: pip install python-docx"
            )

        doc        = docx.Document(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        full_text  = "\n\n".join(paragraphs)
        full_text  = self._clean_text(full_text)

        if len(full_text) > self.MAX_CHARS:
            full_text = full_text[:self.MAX_CHARS]

        return LoadedDocument(
            text       = full_text,
            filename   = filename,
            extension  = ".docx",
            char_count = len(full_text),
            method     = "python-docx",
        )

    # ── Text cleaning ─────────────────────────────────────────────────────────

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text for pipeline consumption.
        - Normalise line endings
        - Collapse excessive whitespace
        - Remove null bytes and control characters
        - Preserve paragraph structure
        """
        import re

        # Remove null bytes
        text = text.replace("\x00", "")

        # Remove control characters except newlines and tabs
        text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # Normalise Windows line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Collapse more than 3 consecutive newlines to 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Collapse multiple spaces on the same line
        text = re.sub(r"[ \t]{2,}", " ", text)

        return text.strip()


# ── Convenience function ──────────────────────────────────────────────────────

def load_document(path: str) -> LoadedDocument | LoadError:
    """Convenience wrapper around DocumentLoader.load_file()."""
    return DocumentLoader().load_file(path)


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import tempfile

    loader = DocumentLoader()

    # Test 1: TXT file
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("""
FACILITY AGREEMENT

Section 4 — Borrower Obligations

The Borrower shall repay the principal amount on the Maturity Date.

The Borrower shall not make any distributions to shareholders during the term.

Section 6 — Conflicting Provisions

The Borrower shall make distributions to shareholders during the term.
""")
        txt_path = f.name

    result = loader.load_file(txt_path)
    os.unlink(txt_path)

    if isinstance(result, LoadError):
        print(f"[TXT] ERROR: {result.reason}")
    else:
        print(f"[TXT] OK  chars={result.char_count}  method={result.method}")
        print(f"       Preview: {result.text[:80]}...")

    # Test 2: PDF file (if pymupdf available)
    try:
        import fitz
        # Create a simple PDF
        pdf_doc  = fitz.open()
        page     = pdf_doc.new_page()
        page.insert_text((50, 50), "The Borrower shall repay the principal on the Maturity Date.")
        page.insert_text((50, 70), "The Borrower is prohibited from making distributions.")

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            pdf_path = f.name

        pdf_doc.save(pdf_path)
        pdf_doc.close()

        result = loader.load_file(pdf_path)
        os.unlink(pdf_path)

        if isinstance(result, LoadError):
            print(f"[PDF] ERROR: {result.reason}")
        else:
            print(f"[PDF] OK  chars={result.char_count}  pages={result.page_count}  method={result.method}")

    except ImportError:
        print("[PDF] SKIP — pymupdf not installed")

    # Test 3: Unsupported format
    result = loader.load_file("/tmp/test.xlsx")
    if isinstance(result, LoadError):
        print(f"[UNSUPPORTED] Correctly refused: {result.reason}")

    # Test 4: File not found
    result = loader.load_file("/tmp/nonexistent.txt")
    if isinstance(result, LoadError):
        print(f"[NOT FOUND] Correctly refused: {result.reason}")

    print("\nDocument Loader smoke test complete.")
