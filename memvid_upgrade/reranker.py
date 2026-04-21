from sentence_transformers import CrossEncoder
from typing import List, Tuple, Optional
import numpy as np

class Reranker:
    """Two-stage reranker:
    - Stage 3: ColBERT MaxSim via BGE-M3 token vectors (fast, ~2ms per candidate)
    - Stage 4: Cross-encoder via bge-reranker-v2-m3 (slow, highest precision)
    """

    def __init__(self, model_name: str = 'BAAI/bge-reranker-v2-m3'):
        print("Loading cross-encoder reranker...")
        self.cross_encoder = CrossEncoder(model_name)
        print("Reranker loaded.")

    def colbert_rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float]],
        embedder,
        top_k: int = 50,
    ) -> List[Tuple[str, float]]:
        """Stage 3: ColBERT MaxSim reranking using BGE-M3 token vectors.
        
        Fast — uses token-level interaction instead of full cross-encoder.
        Apply to top 200 candidates from RRF fusion.
        """
        if not candidates:
            return []

        texts = [c[0] for c in candidates]

        # Embed query with ColBERT vectors
        q_emb = embedder.embed_query(query)
        if q_emb.colbert is None:
            # Fallback to original scores if ColBERT not available
            return candidates[:top_k]

        # Embed all candidates with ColBERT vectors
        d_embs = embedder.embed(texts, return_colbert=True)
        if d_embs.colbert is None:
            return candidates[:top_k]

        # MaxSim scoring
        scores = embedder.colbert_score(
            q_emb.colbert[0],
            d_embs.colbert
        )

        ranked = sorted(
            zip(texts, scores),
            key=lambda x: x[1],
            reverse=True
        )
        return ranked[:top_k]

    def cross_encoder_rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float]],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Stage 4: Full query-document cross-encoder scoring.
        
        Highest precision but slower (~30ms per pair).
        Only apply to top 50 candidates from Stage 3.
        """
        if not candidates:
            return []

        pairs = [(query, c[0]) for c in candidates]
        scores = self.cross_encoder.predict(pairs)

        ranked = sorted(
            zip([c[0] for c in candidates], scores),
            key=lambda x: x[1],
            reverse=True
        )
        return ranked[:top_k]


# Singleton
_reranker: Optional[Reranker] = None

def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
