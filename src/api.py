
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
import json
from .rag import LegalRAG
from .schemas import AskRequest, AskResponse, HealthResponse
from .observability import metrics_response, observe_request

app = FastAPI(title="Egyptian Legal RAG", version="0.1.0")
rag = None


def get_rag() -> LegalRAG:
    global rag
    if rag is None:
        rag = LegalRAG()
    return rag


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    from .vector_store import VectorStore
    store = VectorStore()
    return HealthResponse(
        status="healthy",
        documents_indexed=store.count(),
    )


@app.get("/metrics")
async def metrics() -> Response:
    return Response(metrics_response(), media_type="text/plain")


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    async with observe_request():
        try:
            result = await get_rag().ask(request.question)
            return AskResponse(**result)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="RAG service error") from exc


@app.post("/ask/stream")
async def ask_stream(request: AskRequest):
    """NDJSON streaming endpoint for progressive curl output."""
    try:
        result = await get_rag().ask(request.question)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    async def iterator():
        yield json.dumps({"type": "answer", "text": result["answer"]}, ensure_ascii=False) + "\n"
        yield json.dumps({"type": "sources", "sources": result["sources"]}, ensure_ascii=False) + "\n"

    return StreamingResponse(iterator(), media_type="application/x-ndjson")
