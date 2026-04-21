import faiss
import numpy as np
import pickle
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional

from memvid_upgrade.embedder import BGEm3Embedder
from memvid_upgrade.bm25_index import BM25Index
from memvid_upgrade.reranker import Reranker
from memvid_upgrade.session import SessionMemory


def _rrf(
    ranked_lists: List[List[Tuple[int, float]]],
    k: float = 60.0
) -> List[Tuple[int, float]]:
    """Reciprocal Rank Fusion — rank-based, no score normalization needed."""
    fused = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (chunk_id, _) in enumerate(ranked):
            fused[chunk_id] += 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def _mmr(
    query_vec: np.ndarray,
    candidate_vecs: np.ndarray,
    candidate_ids: List[int],
    top_k: int,
    lambda_: float = 0.7,
) -> List[int]:
    """Maximal Marginal Relevance — balances relevance vs diversity."""
    selected = []
    remaining = list(range(len(candidate_ids)))

    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    candidate_vecs = candidate_vecs / (
        np.linalg.norm(candidate_vecs, axis=1, keepdims=True) + 1e-9
    )

    while len(selected) < top_k and remaining:
        if not selected:
            # First pick: most relevant to query
            scores = candidate_vecs[remaining] @ query_vec
            best = remaining[int(scores.argmax())]
        else:
            rel_scores = candidate_vecs[remaining] @ query_vec
            sel_vecs = candidate_vecs[selected]
            sim_scores = (candidate_vecs[remaining] @ sel_vecs.T).max(axis=1)
            scores = lambda_ * rel_scores - (1 - lambda_) * sim_scores
            best = remaining[int(scores.argmax())]

        selected.append(best)
        remaining.remove(best)

    return [candidate_ids[i] for i in selected]


class HybridRetriever:
    """4-stage retrieval pipeline:
    
    Stage 1: Parallel BM25 + dense FAISS search
    Stage 2: RRF fusion + MMR diversity
    Stage 3: ColBERT late interaction rerank
    Stage 4: Cross-encoder rerank (precise mode only)
    """

    def __init__(self, index_path: str, config: dict = None):
        self.config = config or {}
        self.chunks: List[str] = []
        self.chunk_langs: List[str] = []
        self.translation_groups: List[str] = []

        # Components
        self.embedder = BGEm3Embedder.get_instance()
        self.bm25 = BM25Index()
        self.faiss_index = None
        self.reranker = None  # lazy loaded

        self.index_path = index_path
        if os.path.exists(index_path + '.faiss'):
            self.load(index_path)

    def build(
        self,
        chunks: List[dict],  # [{'text': str, 'lang': str, 'translation_group': str}]
    ):
        """Build all indexes from a list of chunk dicts."""
        print(f"Building indexes for {len(chunks)} chunks...")
        self.chunks = [c['text'] for c in chunks]
        self.chunk_langs = [c.get('lang', 'en') for c in chunks]
        self.translation_groups = [c.get('translation_group', '') for c in chunks]

        # Embed all chunks
        print("Embedding chunks...")
        result = self.embedder.embed(self.chunks, batch_size=16)
        embeddings = result.dense.astype('float32')

        # Build FAISS index
        print("Building FAISS index...")
        self.faiss_index = self._build_faiss(embeddings)

        # Build BM25 index
        self.bm25.add(self.chunks, chunk_ids=list(range(len(self.chunks))))

        print("All indexes built.")

    def _build_faiss(self, embeddings: np.ndarray) -> faiss.Index:
        n, dim = embeddings.shape
        faiss.normalize_L2(embeddings)

        if n < 10_000:
            index = faiss.IndexFlatIP(dim)
        elif n < 1_000_000:
            nlist = min(1024, n // 10)
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(
                quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT
            )
            index.train(embeddings)
            index.nprobe = 64
        else:
            nlist = 4096
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFPQ(quantizer, dim, nlist, 64, 8)
            index.train(embeddings)
            index.nprobe = 128

        index.add(embeddings)
        return index

    def _colbert_rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float]],
        top_k: int = 50,
    ) -> List[Tuple[str, float]]:
        """ColBERT MaxSim reranking using BGE-M3 token vectors.
        
        Uses the already-loaded embedder — no extra model needed.
        Much faster than cross-encoder since it reuses existing embeddings.
        """
        if not candidates:
            return []

        texts = [c[0] for c in candidates]

        # Single BGE-M3 call for query with ColBERT vectors
        q_emb = self.embedder.embed_query(query)
        if q_emb.colbert is None:
            return candidates[:top_k]

        # Single BGE-M3 call for all candidates together (batched)
        d_embs = self.embedder.embed(texts, batch_size=16, return_colbert=True)
        if d_embs.colbert is None:
            return candidates[:top_k]

        # MaxSim scoring
        scores = self.embedder.colbert_score(q_emb.colbert[0], d_embs.colbert)
        ranked = sorted(zip(texts, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def _dense_search(
        self,
        query: str,
        k: int,
        query_vec: np.ndarray = None,  # pass pre-computed vec to avoid re-embedding
    ) -> tuple:
        """Returns (results, query_vec) so callers can reuse the query embedding.
        
        query_vec is cached and passed back so MMR and ColBERT never need
        to re-embed the query — one BGE-M3 call serves the entire pipeline.
        """
        if query_vec is None:
            q_emb = self.embedder.embed_query(query)
            query_vec = q_emb.dense.astype('float32')

        vec = query_vec.copy()
        faiss.normalize_L2(vec)
        distances, indices = self.faiss_index.search(vec, k)
        results = [
            (int(indices[0][i]), float(distances[0][i]))
            for i in range(k)
            if indices[0][i] != -1
        ]
        return results, query_vec

    def _filter_by_lang(
        self,
        candidates: List[Tuple[str, float]],
        lang: str
    ) -> List[Tuple[str, float]]:
        return [
            (text, score) for text, score in candidates
            if self.chunk_langs[self.chunks.index(text)] == lang
        ]

    def search(
        self,
        query: str,
        top_k: int = 10,
        pipeline: str = 'hybrid',   # 'fast' | 'hybrid' | 'precise'
        lang: Optional[str] = None,
        session: Optional[SessionMemory] = None,
    ) -> List[Tuple[str, float]]:

        first_k = self.config.get('first_stage_k', 20)

        # ── Pre-compute query embedding ONCE ─────────────────────
        # This single BGE-M3 call serves Stage 1 (dense search),
        # Stage 2 (MMR), and Stage 3 (ColBERT) — no re-embedding anywhere.
        q_emb = self.embedder.embed_query(query)
        query_vec = q_emb.dense.astype('float32')

        # ── Stage 1: First-stage recall ──────────────────────────
        print(f"Stage 1: {pipeline} retrieval...")
        if pipeline == 'fast':
            bm25_results = self.bm25.search(query, first_k)
            candidates_ids = bm25_results
        else:
            with ThreadPoolExecutor(max_workers=2) as ex:
                # Pass pre-computed query_vec so dense search skips re-embedding
                bm25_fut  = ex.submit(self.bm25.search, query, first_k)
                dense_fut = ex.submit(self._dense_search, query, first_k, query_vec)
                bm25_res        = bm25_fut.result()
                dense_res, _    = dense_fut.result()   # unpack (results, vec) tuple
            candidates_ids = _rrf([bm25_res, dense_res])

        # Decode chunk texts
        candidates = [
            (self.chunks[cid], score)
            for cid, score in candidates_ids
            if cid < len(self.chunks)
        ]

        # ── Language filter ───────────────────────────────────────
        if lang:
            candidates = self._filter_by_lang(candidates, lang)

        # ── Stage 2: MMR diversity ────────────────────────────────
        # Uses pre-computed query_vec and FAISS stored vectors — zero extra embedding calls
        print(f"Stage 2: MMR diversity on {len(candidates)} candidates...")
        if len(candidates) > top_k:
            texts = [c[0] for c in candidates]

            # Look up pre-stored FAISS vectors by chunk ID instead of re-embedding
            # This is pure numpy dot product — runs in microseconds not seconds
            chunk_ids = [self.chunks.index(t) for t in texts if t in self.chunks]
            if len(chunk_ids) == len(texts):
                # Reconstruct vectors from FAISS index directly
                stored_vecs = np.zeros((len(chunk_ids), self.faiss_index.d), dtype='float32')
                self.faiss_index.reconstruct_batch(chunk_ids, stored_vecs)
                q_vec = query_vec[0] if query_vec.ndim > 1 else query_vec
                ids = _mmr(q_vec, stored_vecs, list(range(len(texts))), top_k=min(20, len(candidates)))
            else:
                # Fallback: embed candidates (only if FAISS reconstruct fails)
                vecs = self.embedder.embed(texts).dense.astype('float32')
                q_vec = query_vec[0] if query_vec.ndim > 1 else query_vec
                ids = _mmr(q_vec, vecs, list(range(len(texts))), top_k=min(20, len(candidates)))

            candidates = [(texts[i], candidates[i][1]) for i in ids]

        # ── Stage 3: ColBERT (precise only) ─────────────────────
        # ColBERT token-vector reranking is GPU-optimized and too slow on CPU
        # for interactive chatbot use. Reserved for explicit precise mode only.
        if pipeline == 'precise':
            print(f"Stage 3: ColBERT rerank on {len(candidates)} candidates...")
            candidates = self._colbert_rerank(query, candidates, top_k=50)

        # ── Stage 4: Cross-encoder (precise only) ─────────────────
        # Only loads bge-reranker-v2-m3 when pipeline='precise'
        if pipeline == 'precise':
            print(f"Stage 4: Cross-encoder rerank on {len(candidates)} candidates...")
            if self.reranker is None:
                from memvid_upgrade.reranker import get_reranker
                self.reranker = get_reranker()
            candidates = self.reranker.cross_encoder_rerank(
                query, candidates, top_k=top_k
            )
        results = candidates[:top_k]

        # ── Session memory ─────────────────────────────────────────
        if session:
            results = session.apply(results)
            session.record(results)

        return results

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        faiss.write_index(self.faiss_index, path + '.faiss')
        self.bm25.save(path + '.bm25')
        with open(path + '.meta', 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'chunk_langs': self.chunk_langs,
                'translation_groups': self.translation_groups,
            }, f)
        print(f"Index saved to {path}")

    def load(self, path: str):
        self.faiss_index = faiss.read_index(path + '.faiss')
        self.bm25.load(path + '.bm25')
        with open(path + '.meta', 'rb') as f:
            data = pickle.load(f)
        self.chunks = data['chunks']
        self.chunk_langs = data['chunk_langs']
        self.translation_groups = data['translation_groups']
        print(f"Index loaded from {path}")
