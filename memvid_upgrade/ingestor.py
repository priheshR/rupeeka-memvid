import fitz  # pymupdf
import httpx
import uuid
import os
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from memvid_upgrade.lang_detect import detect_language
from memvid_upgrade.translator import get_translator


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
        i += chunk_size - overlap
    return chunks


def extract_pdf(path: str) -> str:
    """Extract all text from a PDF file."""
    doc = fitz.open(path)
    pages = []
    for page in doc:
        text = page.get_text().strip()
        if text:
            pages.append(text)
    doc.close()
    return '\n\n'.join(pages)


def extract_url(url: str) -> str:
    """Scrape main text content from a URL."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = httpx.get(url, headers=headers, timeout=30, follow_redirects=True)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')

    # Remove noise
    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
        tag.decompose()

    # Try to get main content first
    main = soup.find('main') or soup.find('article') or soup.find('body')
    text = main.get_text(separator='\n') if main else soup.get_text(separator='\n')

    # Clean up whitespace
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return '\n'.join(lines)


class Ingestor:
    """Handles PDF, URL, and raw text ingestion.
    
    For each source:
    1. Extract raw text
    2. Chunk into overlapping segments
    3. Detect source language
    4. Translate each chunk to Sinhala + Tamil via Gemini
    5. Return all chunks ready for indexing
    """

    def __init__(
        self,
        target_langs: List[str] = None,
        chunk_size: int = 400,
        overlap: int = 50,
    ):
        self.target_langs = target_langs or ['si', 'ta']
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.translator = get_translator(target_langs=self.target_langs)

    def _process_text(
        self,
        text: str,
        source: str,
        metadata: dict = None,
    ) -> List[Dict]:
        """Chunk, detect language, translate, and return chunk dicts."""
        chunks = chunk_text(text, self.chunk_size, self.overlap)
        print(f"  → {len(chunks)} chunks from {source}")

        all_chunks = []
        for i, chunk in enumerate(chunks):
            group_id = str(uuid.uuid4())
            src_lang = detect_language(chunk)
            translations = self.translator.translate_all(chunk, source_lang=src_lang)

            for lang, translated_text in translations.items():
                all_chunks.append({
                    'text': translated_text,
                    'lang': lang,
                    'translation_group': group_id,
                    'source': source,
                    'chunk_index': i,
                    'is_canonical': lang == src_lang,
                    'metadata': metadata or {},
                })

        return all_chunks

    def ingest_text(
        self,
        text: str,
        source: str = 'custom',
        metadata: dict = None,
    ) -> List[Dict]:
        """Ingest raw text."""
        print(f"Ingesting text ({len(text)} chars)...")
        return self._process_text(text, source, metadata)

    def ingest_pdf(
        self,
        path: str,
        metadata: dict = None,
    ) -> List[Dict]:
        """Ingest a PDF file."""
        print(f"Ingesting PDF: {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"PDF not found: {path}")
        text = extract_pdf(path)
        print(f"  → Extracted {len(text)} characters from PDF")
        source = os.path.basename(path)
        return self._process_text(text, source, metadata)

    def ingest_url(
        self,
        url: str,
        metadata: dict = None,
    ) -> List[Dict]:
        """Ingest content from a URL."""
        print(f"Ingesting URL: {url}")
        text = extract_url(url)
        print(f"  → Extracted {len(text)} characters from URL")
        return self._process_text(text, url, metadata)
