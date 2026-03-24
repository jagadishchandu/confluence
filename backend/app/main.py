from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .schemas import AskRequest, AskResponse
from .ingest import sync_confluence
from .llm_graph import GRAPH
 
app = FastAPI(title="RAG Chart Bot API (FastAPI + LangGraph + Bedrock)")
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.UI_ORIGIN],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
@app.get("/health")
def health():
    return {"ok": True}
 
@app.post("/sync")
def sync(limit: int = 25):
    return sync_confluence(limit=limit)
 
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    out = GRAPH.invoke({"question": req.question, "want_chart": req.want_chart, "filters": req.filters})
    return AskResponse(answer=out["answer"], citations=out["citations"], chart=out.get("chart"))