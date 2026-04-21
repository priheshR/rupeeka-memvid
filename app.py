import os
import uuid
import shutil
import threading
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from memvid_upgrade.knowledge_base import KnowledgeBase
from memvid_upgrade.session import SessionMemory

app = FastAPI(title="Memvid Knowledge Base API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared KB instance ───────────────────────────────────
kb = KnowledgeBase(
    name="main_kb",
    data_dir="./kb_data",
    target_langs=["si", "ta"]
)

# ── Session store (in-memory) ────────────────────────────
sessions: dict[str, SessionMemory] = {}

# ── Background build ─────────────────────────────────────
_build_lock = threading.Lock()
_is_building = False

def _trigger_build():
    """Run kb.build() in a background thread — returns immediately."""
    global _is_building

    def _build():
        global _is_building
        if not _build_lock.acquire(blocking=False):
            print("Build already running, skipping.")
            return
        try:
            _is_building = True
            print("Background build started...")
            kb.build()
            print("Background build complete.")
        except Exception as e:
            print(f"Background build error: {e}")
        finally:
            _is_building = False
            _build_lock.release()

    threading.Thread(target=_build, daemon=True).start()


def get_session(session_id: Optional[str]) -> tuple[str, SessionMemory]:
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = SessionMemory()
    return session_id, sessions[session_id]


# ── Request / Response models ────────────────────────────

class TagsModel(BaseModel):
    date: str = ""
    key_areas: list = []
    keywords: list = []
    doc_type: str = ""
    author: str = ""

class IngestTextRequest(BaseModel):
    text: str
    source: str = "custom"
    metadata: dict = {}
    tags: TagsModel = TagsModel()

class IngestUrlRequest(BaseModel):
    url: str
    metadata: dict = {}
    tags: TagsModel = TagsModel()

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    pipeline: str = "hybrid"
    lang: Optional[str] = None
    session_id: Optional[str] = None

class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    response_lang: str = "en"
    session_id: Optional[str] = None


# ── Routes ───────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "Memvid Knowledge Base API"}


@app.get("/status")
def status():
    return {
        "status": "ok",
        "stats": kb.stats(),
        "active_sessions": len(sessions),
        "building": _is_building,
    }


@app.post("/ingest/text")
def ingest_text(req: IngestTextRequest):
    try:
        count = kb.ingest_text(
            req.text,
            source=req.source,
            metadata=req.metadata,
            tags=req.tags.dict()
        )
        _trigger_build()
        return {"status": "ok", "chunks_added": count, "building": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/pdf")
async def ingest_pdf(
    file: UploadFile = File(...),
    date: str = "",
    key_areas: str = "",
    keywords: str = "",
    doc_type: str = "",
    author: str = "",
):
    try:
        tmp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        tags = {
            "date":      date,
            "key_areas": [x.strip() for x in key_areas.split(",") if x.strip()],
            "keywords":  [x.strip() for x in keywords.split(",") if x.strip()],
            "doc_type":  doc_type,
            "author":    author,
        }
        count = kb.ingest_pdf(
            tmp_path,
            metadata={"filename": file.filename},
            tags=tags
        )
        _trigger_build()
        os.remove(tmp_path)
        return {"status": "ok", "chunks_added": count, "filename": file.filename, "building": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/url")
def ingest_url(req: IngestUrlRequest):
    try:
        count = kb.ingest_url(
            req.url,
            metadata=req.metadata,
            tags=req.tags.dict()
        )
        _trigger_build()
        return {"status": "ok", "chunks_added": count, "url": req.url, "building": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
def search(req: SearchRequest):
    try:
        session_id, session = get_session(req.session_id)
        results = kb.search(
            req.query,
            top_k=req.top_k,
            pipeline=req.pipeline,
            lang=req.lang,
            session=session,
        )
        return {
            "results": results,
            "session_id": session_id,
            "total": len(results),
        }
    except RuntimeError as e:
        return {"results": [], "session_id": req.session_id or "", "total": 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
def ask(req: AskRequest):
    try:
        session_id, session = get_session(req.session_id)
        result = kb.ask(
            req.question,
            top_k=req.top_k,
            session=session,
            response_lang=req.response_lang,
        )
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "session_id": session_id,
            "response_lang": req.response_lang,
        }
    except RuntimeError as e:
        return {
            "answer": str(e),
            "sources": [],
            "session_id": req.session_id or "",
            "response_lang": req.response_lang,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/factcheck")
async def factcheck(request: Request):
    """Stream fact-check results as Server-Sent Events."""
    body = await request.json()
    text = body.get("text", "")
    source = body.get("source", "")

    if not text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    from memvid_upgrade.factchecker import get_factchecker
    fc = get_factchecker(kb=kb)

    def event_stream():
        for event in fc.analyse_stream(text, source=source):
            yield event

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        return {"status": "cleared"}
    return {"status": "not_found"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
