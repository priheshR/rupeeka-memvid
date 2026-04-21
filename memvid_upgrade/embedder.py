from FlagEmbedding import BGEM3FlagModel
from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np

@dataclass
class EmbeddingResult:
    dense: np.ndarray                    # shape (N, 1024) — for FAISS
    sparse: List[Dict[int, float]]       # token_id → weight — for BM25-like lookup
    colbert: Optional[List] = None       # token-level vecs — for late interaction

class BGEm3Embedder:
    """Wraps BAAI/bge-m3 for dense + sparse + ColBERT output.
    Singleton — only loaded once, shared across encoder and retriever.
    """
    _instance = None

    def __init__(self, use_fp16: bool = False):
        print("Loading BGE-M3 model...")
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=use_fp16)
        print("BGE-M3 loaded.")

    @classmethod
    def get_instance(cls) -> 'BGEm3Embedder':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def embed(
        self,
        texts: List[str],
        batch_size: int = 16,
        return_colbert: bool = False,
    ) -> EmbeddingResult:
        output = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=8192,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=return_colbert,
        )
        return EmbeddingResult(
            dense=output['dense_vecs'],
            sparse=output['lexical_weights'],
            colbert=output.get('colbert_vecs'),
        )

    def embed_query(self, query: str) -> EmbeddingResult:
        """Single query — always include ColBERT for re-ranking."""
        return self.embed([query], return_colbert=True)

    def colbert_score(
        self,
        query_vecs: List[np.ndarray],
        doc_vecs_list: List[List[np.ndarray]]
    ) -> List[float]:
        """MaxSim ColBERT scoring — query token vs doc token matrix."""
        scores = []
        for doc_vecs in doc_vecs_list:
            Q = np.array(query_vecs)    # (q_len, dim)
            D = np.array(doc_vecs)      # (d_len, dim)
            sim = Q @ D.T               # (q_len, d_len)
            score = sim.max(axis=1).sum()  # MaxSim then sum
            scores.append(float(score))
        return scores
