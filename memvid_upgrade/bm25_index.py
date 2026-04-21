from rank_bm25 import BM25Okapi
from typing import List, Tuple
import re
import json
import pickle

def tokenize(text: str) -> List[str]:
    """Lowercase + strip punctuation tokenizer."""
    return re.sub(r'[^\w\s]', '', text.lower()).split()

class BM25Index:
    """Lightweight BM25 index that sits alongside FAISS for hybrid search."""

    def __init__(self):
        self.bm25 = None
        self.corpus_tokens: List[List[str]] = []
        self.chunk_ids: List[int] = []   # maps BM25 rank → original chunk index

    def add(self, texts: List[str], chunk_ids: List[int]):
        """Build BM25 index from a list of texts."""
        self.corpus_tokens = [tokenize(t) for t in texts]
        self.chunk_ids = chunk_ids
        self.bm25 = BM25Okapi(self.corpus_tokens)
        print(f"BM25 index built: {len(texts)} documents")

    def search(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        """Returns [(chunk_id, score), ...] sorted by BM25 score descending."""
        if self.bm25 is None:
            return []
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = scores.argsort()[::-1][:top_k]
        return [
            (self.chunk_ids[i], float(scores[i]))
            for i in top_idx
            if scores[i] > 0
        ]

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'corpus_tokens': self.corpus_tokens,
                'chunk_ids': self.chunk_ids,
            }, f)
        print(f"BM25 index saved to {path}")

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.corpus_tokens = data['corpus_tokens']
        self.chunk_ids = data['chunk_ids']
        self.bm25 = BM25Okapi(self.corpus_tokens)
        print(f"BM25 index loaded from {path}")
