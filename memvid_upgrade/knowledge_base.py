import os
import json
import pickle
from typing import List, Dict, Optional
from memvid import MemvidEncoder, MemvidRetriever

from memvid_upgrade.ingestor import Ingestor
from memvid_upgrade.retriever import HybridRetriever
from memvid_upgrade.session import SessionMemory


class KnowledgeBase:
    """Combines memvid storage with hybrid retrieval.
    
    Memvid handles the .mp4 memory file (persistent, portable storage).
    HybridRetriever handles BGE-M3 + BM25 + reranking on top of it.
    
    Usage:
        kb = KnowledgeBase('my_kb')
        kb.ingest_text('Some content...')
        kb.ingest_pdf('document.pdf')
        kb.ingest_url('https://example.com')
        kb.build()
        results = kb.search('my query')
    """

    def __init__(
        self,
        name: str,
        data_dir: str = './kb_data',
        target_langs: List[str] = None,
    ):
        self.name = name
        self.data_dir = data_dir
        self.target_langs = target_langs or ['si', 'ta']

        # File paths
        os.makedirs(data_dir, exist_ok=True)
        self.video_path  = os.path.join(data_dir, f'{name}.mp4')
        self.index_path  = os.path.join(data_dir, f'{name}_index')
        self.chunks_path = os.path.join(data_dir, f'{name}_chunks.pkl')
        self.stats_path  = os.path.join(data_dir, f'{name}_stats.json')

        # Components
        self.ingestor = Ingestor(target_langs=self.target_langs)
        self.encoder  = MemvidEncoder()
        self.retriever: Optional[HybridRetriever] = None

        # Pending chunks waiting to be built
        self._pending: List[Dict] = []
        self._stats = {
            'total_sources': 0,
            'total_chunks': 0,
            'languages': list(set(['en'] + self.target_langs)),
            'sources': [],
        }

        # Load existing stats if available
        if os.path.exists(self.stats_path):
            with open(self.stats_path) as f:
                self._stats = json.load(f)

    # ── Ingestion ────────────────────────────────────────────

    def _build_metadata(self, base: dict, tags: dict) -> dict:
        """Merge base metadata with user-supplied tags."""
        from datetime import datetime
        return {
            **(base or {}),
            'date':      tags.get('date') or datetime.today().strftime('%Y-%m-%d'),
            'key_areas': tags.get('key_areas', []),   # list of strings
            'keywords':  tags.get('keywords', []),     # list of strings
            'doc_type':  tags.get('doc_type', ''),     # e.g. "report", "article"
            'author':    tags.get('author', ''),
        }

    def ingest_text(self, text: str, source: str = 'custom',
                    metadata: dict = None, tags: dict = None):
        """Add raw text to the knowledge base."""
        meta = self._build_metadata(metadata, tags or {})
        chunks = self.ingestor.ingest_text(text, source, meta)
        self._pending.extend(chunks)
        self._stats['total_sources'] += 1
        self._stats['sources'].append({
            'type': 'text', 'source': source, 'tags': meta
        })
        print(f"Queued {len(chunks)} chunks from text")
        return len(chunks)

    def ingest_pdf(self, path: str, metadata: dict = None, tags: dict = None):
        """Add a PDF file to the knowledge base."""
        meta = self._build_metadata(metadata, tags or {})
        chunks = self.ingestor.ingest_pdf(path, meta)
        self._pending.extend(chunks)
        self._stats['total_sources'] += 1
        self._stats['sources'].append({
            'type': 'pdf', 'source': path, 'tags': meta
        })
        print(f"Queued {len(chunks)} chunks from PDF")
        return len(chunks)

    def ingest_url(self, url: str, metadata: dict = None, tags: dict = None):
        """Add web content to the knowledge base."""
        meta = self._build_metadata(metadata, tags or {})
        chunks = self.ingestor.ingest_url(url, meta)
        self._pending.extend(chunks)
        self._stats['total_sources'] += 1
        self._stats['sources'].append({
            'type': 'url', 'source': url, 'tags': meta
        })
        print(f"Queued {len(chunks)} chunks from URL")
        return len(chunks)

    # ── Build ────────────────────────────────────────────────

    def _load_existing_chunks(self) -> list:
        """Load previously indexed chunks from disk."""
        if os.path.exists(self.chunks_path):
            with open(self.chunks_path, 'rb') as f:
                return pickle.load(f)
        return []

    def build(self):
        """Build memvid memory file + hybrid indexes from ALL chunks.

        Merges new pending chunks with existing indexed chunks so
        nothing is lost across successive builds.
        """
        if not self._pending:
            print("No pending chunks to build.")
            return

        # Load existing chunks and merge with new ones
        existing = self._load_existing_chunks()
        all_chunks = existing + self._pending

        print(f"\nBuilding knowledge base '{self.name}'...")
        print(f"  Existing chunks : {len(existing)}")
        print(f"  New chunks      : {len(self._pending)}")
        print(f"  Total chunks    : {len(all_chunks)}")

        # Rebuild memvid encoder from scratch with all chunks
        self.encoder = MemvidEncoder()
        for chunk in all_chunks:
            self.encoder.add_text(
                chunk['text'],
                chunk_size=10000,
                overlap=0,
            )

        # Build memvid .mp4 memory file
        print("Building memvid memory file...")
        self.encoder.build_video(self.video_path, self.index_path)

        # Build hybrid retriever on top of all chunks
        print("Building hybrid retriever...")
        self.retriever = HybridRetriever(self.index_path + '_hybrid')
        self.retriever.build(all_chunks)
        self.retriever.save(self.index_path + '_hybrid')

        # Save ALL chunks to disk
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(all_chunks, f)

        # Update and save stats
        self._stats['total_chunks'] = len(all_chunks)
        with open(self.stats_path, 'w') as f:
            json.dump(self._stats, f, indent=2)

        print(f"\nKnowledge base built successfully!")
        print(f"  Memory file : {self.video_path}")
        print(f"  Index       : {self.index_path}")
        print(f"  Total chunks: {len(all_chunks)}")
        print(f"  Languages   : {self.target_langs}")

        self._pending = []

    # ── Search ───────────────────────────────────────────────

    def _ensure_retriever(self):
        """Load retriever if not already loaded."""
        if self.retriever is not None:
            return
        index = self.index_path + '_hybrid'
        if os.path.exists(index + '.faiss'):
            print(f"Loading existing index from {index}...")
            self.retriever = HybridRetriever(index)
        else:
            raise RuntimeError(
                "Knowledge base is empty. Please add content first using "
                "the sidebar before asking questions."
            )

    def search(
        self,
        query: str,
        top_k: int = 5,
        pipeline: str = 'hybrid',
        lang: Optional[str] = None,
        session: Optional[SessionMemory] = None,
    ) -> List[Dict]:
        """Search the knowledge base."""
        self._ensure_retriever()
        results = self.retriever.search(
            query,
            top_k=top_k,
            pipeline=pipeline,
            lang=lang,
            session=session,
        )
        # Attach metadata to each result
        out = []
        for text, score in results:
            meta = {}
            if text in self.retriever.chunks:
                idx = self.retriever.chunks.index(text)
                if idx < len(self.retriever.chunk_langs):
                    # Load full metadata from saved chunks
                    try:
                        import pickle
                        with open(self.chunks_path, 'rb') as f:
                            all_chunks = pickle.load(f)
                        if idx < len(all_chunks):
                            meta = all_chunks[idx].get('metadata', {})
                    except Exception:
                        pass
            out.append({'text': text, 'score': round(score, 4), 'metadata': meta})
        return out

    def ask(
        self,
        question: str,
        top_k: int = 5,
        session: Optional[SessionMemory] = None,
        response_lang: str = 'en',
    ) -> Dict:
        """Search + generate answer using Gemini."""
        from google import genai

        # Retrieve relevant chunks
        results = self.search(question, top_k=top_k, session=session)
        context = '\n\n'.join([r['text'] for r in results])

        lang_names = {'en': 'English', 'si': 'Sinhala', 'ta': 'Tamil'}
        respond_in = lang_names.get(response_lang, 'English')

        prompt = f"""Answer the question using only the context provided below.
If the answer is not in the context, say you don't know.
Respond in {respond_in}.

Context:
{context}

Question: {question}

Answer:"""

        client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=prompt
        )

        return {
            'answer': response.text.strip(),
            'sources': results,
            'response_lang': response_lang,
        }

    def stats(self) -> Dict:
        """Return knowledge base statistics."""
        return self._stats
