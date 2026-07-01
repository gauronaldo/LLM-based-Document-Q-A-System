"""Load PDF, TXT, and DOCX documents with source metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}
SCANNED_PDF_MESSAGE = (
    "No extractable text was found. This PDF may be scanned. "
    "OCR support is not included in the current version."
)


class DocumentLoaderError(Exception):
    """Base error raised by the document loader."""


class UnsupportedFileTypeError(DocumentLoaderError):
    """Raised when the uploaded file type is not supported."""


class EmptyDocumentError(DocumentLoaderError):
    """Raised when a document contains no extractable text."""


class DocumentLoader:
    """Extract text from supported document types while preserving metadata."""

    def load(self, file_path: str, file_name: str, file_id: str) -> list[dict[str, Any]]:
        """Load a document and return page-level or section-level text objects."""

        path = Path(file_path)
        extension = path.suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise UnsupportedFileTypeError(
                f"Unsupported file type '{extension or 'unknown'}'. "
                f"Supported file types are: {supported}."
            )

        if extension == ".pdf":
            return self.load_pdf(file_path, file_name, file_id)
        if extension == ".txt":
            return self.load_txt(file_path, file_name, file_id)
        return self.load_docx(file_path, file_name, file_id)

    def load_pdf(
        self,
        file_path: str,
        file_name: str,
        file_id: str,
    ) -> list[dict[str, Any]]:
        """Extract PDF text page by page."""

        documents: list[dict[str, Any]] = []

        try:
            import fitz

            with fitz.open(file_path) as pdf:
                for page_index, page in enumerate(pdf, start=1):
                    text = page.get_text("text").strip()
                    if not text:
                        continue

                    documents.append(
                        self._document(
                            text=text,
                            file_id=file_id,
                            file_name=file_name,
                            page=page_index,
                        )
                    )
        except Exception as exc:
            if isinstance(exc, ImportError):
                raise DocumentLoaderError(
                    "PyMuPDF is required to load PDF files. "
                    "Install project dependencies with 'pip install -r requirements.txt'."
                ) from exc
            raise DocumentLoaderError(f"Failed to load PDF '{file_name}': {exc}") from exc

        if not documents:
            raise EmptyDocumentError(SCANNED_PDF_MESSAGE)

        return documents

    def load_txt(
        self,
        file_path: str,
        file_name: str,
        file_id: str,
    ) -> list[dict[str, Any]]:
        """Read a TXT file as a single document object."""

        path = Path(file_path)

        try:
            text = path.read_text(encoding="utf-8-sig").strip()
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception as exc:
            raise DocumentLoaderError(f"Failed to load TXT file '{file_name}': {exc}") from exc

        if not text:
            raise EmptyDocumentError(f"No extractable text was found in '{file_name}'.")

        return [
            self._document(
                text=text,
                file_id=file_id,
                file_name=file_name,
                page=1,
            )
        ]

    def load_docx(
        self,
        file_path: str,
        file_name: str,
        file_id: str,
    ) -> list[dict[str, Any]]:
        """Extract non-empty DOCX paragraphs as document objects."""

        try:
            from docx import Document as DocxDocument

            docx = DocxDocument(file_path)
        except Exception as exc:
            if isinstance(exc, ImportError):
                raise DocumentLoaderError(
                    "python-docx is required to load DOCX files. "
                    "Install project dependencies with 'pip install -r requirements.txt'."
                ) from exc
            raise DocumentLoaderError(f"Failed to load DOCX file '{file_name}': {exc}") from exc

        documents: list[dict[str, Any]] = []
        for paragraph_index, paragraph in enumerate(docx.paragraphs, start=1):
            text = paragraph.text.strip()
            if not text:
                continue

            document = self._document(
                text=text,
                file_id=file_id,
                file_name=file_name,
                page=None,
            )
            document["metadata"]["paragraph_index"] = paragraph_index
            documents.append(document)

        if not documents:
            raise EmptyDocumentError(f"No extractable text was found in '{file_name}'.")

        return documents

    @staticmethod
    def _document(
        text: str,
        file_id: str,
        file_name: str,
        page: int | None,
    ) -> dict[str, Any]:
        """Create the common document object used by the RAG pipeline."""

        return {
            "text": text,
            "metadata": {
                "file_id": file_id,
                "file_name": file_name,
                "page": page,
            },
        }
