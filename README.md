# Memvid Multilingual Knowledge Base

Hybrid RAG system with auto-translation to Sinhala and Tamil.

## Stack
- **BGE-M3** — multilingual embeddings (dense + sparse + ColBERT)
- **BM25 + FAISS** — hybrid first-stage retrieval with RRF fusion  
- **ColBERT** — late interaction reranking (Stage 3)
- **Cross-encoder** — precision reranking (Stage 4, precise mode)
- **Gemini 2.5 Pro** — translation + answer generation
- **FastAPI** — REST API backend
- **Memvid** — .mp4 memory file storage

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GOOGLE_API_KEY=your-key-here
```

## Run
```bash
./start.sh
```
Then open `chatbot.html` in your browser.

## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /status | KB stats + active sessions |
| POST | /ingest/text | Add raw text |
| POST | /ingest/url | Scrape and add URL |
| POST | /ingest/pdf | Upload and add PDF |
| POST | /search | Hybrid search |
| POST | /ask | QA with Gemini answer |
| DELETE | /session/{id} | Clear session memory |

## Pipelines
- `fast` — BM25 only, <10ms
- `hybrid` — BM25 + FAISS + ColBERT, <80ms  
- `precise` — full 4-stage cascade with cross-encoder, <350ms

## File Structure
```
my-memvid-upgrade/
├── app.py                    # FastAPI server
├── chatbot.html              # Browser UI
├── start.sh                  # One-command startup
├── requirements.txt
├── kb_data/                  # Knowledge base files
│   ├── main_kb.mp4           # Memvid memory file
│   ├── main_kb_index.*       # Memvid index
│   └── main_kb_*_hybrid.*    # FAISS + BM25 indexes
└── memvid_upgrade/
    ├── knowledge_base.py     # Main KB class
    ├── ingestor.py           # PDF + URL + text extraction
    ├── retriever.py          # 4-stage hybrid pipeline
    ├── embedder.py           # BGE-M3 wrapper
    ├── bm25_index.py         # BM25 index
    ├── reranker.py           # ColBERT + cross-encoder
    ├── translator.py         # Gemini translation
    ├── lang_detect.py        # Unicode-aware detection
    ├── session.py            # Session memory
    └── config.py             # Configuration
