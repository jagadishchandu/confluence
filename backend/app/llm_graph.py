import json
from typing import Any, Dict, List, Optional, TypedDict
from langgraph.graph import StateGraph, END
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from .config import settings
from .qdrant_store import search as qdrant_search
from .schemas import ChartSpec, Citation
 
 
class GraphState(TypedDict, total=False):
    question: str
    want_chart: bool
    filters: Optional[Dict[str, Any]]
    context: str
    citations: List[Citation]
    answer: str
    chart: Optional[ChartSpec]
 
 
def _chat() -> ChatBedrock:
    return ChatBedrock(model_id=settings.BEDROCK_CHAT_MODEL_ID, region_name=settings.AWS_REGION, model_kwargs={"max_tokens": 900})
 
 
def _embed() -> BedrockEmbeddings:
    return BedrockEmbeddings(model_id=settings.BEDROCK_EMBED_MODEL_ID, region_name=settings.AWS_REGION)
 
 
def _hits_to_context_and_citations(hits) -> (str, List[Citation]):
    ctx, cits = [], []
    for h in hits:
        p = h.payload or {}
        ctx.append(f"[chunk_id={p.get('chunk_id')}] title={p.get('title')} url={p.get('url')} type={p.get('chunk_type')}\n{p.get('text','')}")
        cits.append(Citation(page_id=str(p.get("page_id")), url=str(p.get("url")), title=str(p.get("title")), chunk_id=str(p.get("chunk_id")), chunk_type=str(p.get("chunk_type") or "text")))
    return "\n\n---\n\n".join(ctx), cits
 
 
def node_retrieve(state: GraphState) -> GraphState:
    qvec = _embed().embed_query(state["question"])
    hits = qdrant_search(qvec, k=6, filters=state.get("filters"))
    context, citations = _hits_to_context_and_citations(hits)
    return {"context": context, "citations": citations}
 
 
def node_answer(state: GraphState) -> GraphState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a RAG assistant. Use ONLY the CONTEXT. If insufficient, say what’s missing. Include citations like [chunk_id=...]."),
        ("user", "QUESTION:\n{question}\n\nCONTEXT:\n{context}")
    ])
    out = _chat().invoke(prompt.invoke({"question": state["question"], "context": state["context"]}))
    return {"answer": out.content}
 
 
def node_chart(state: GraphState) -> GraphState:
    if not state.get("want_chart", True):
        return {"chart": None}
 
    parser = PydanticOutputParser(pydantic_object=ChartSpec)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Return a ChartSpec STRICTLY from CONTEXT; do NOT invent numbers. Prefer TABLE chunks. "
         "If no numeric data, return empty data []. Return ONLY JSON.\n\n{format_instructions}"),
        ("user", "REQUEST:\n{question}\n\nCONTEXT:\n{context}\n\nCITATIONS:\n{citations_json}")
    ])
 
    raw = _chat().invoke(prompt.invoke({
        "question": state["question"],
        "context": state["context"],
        "citations_json": json.dumps([c.model_dump() for c in state["citations"]]),
        "format_instructions": parser.get_format_instructions(),
    })).content
 
    spec = parser.parse(raw)
    if not spec.citations:
        spec.citations = state["citations"][:5]
    return {"chart": spec}
 
 
def build_graph():
    g = StateGraph(GraphState)
    g.add_node("retrieve", node_retrieve)
    g.add_node("answer", node_answer)
    g.add_node("chart", node_chart)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", "chart")
    g.add_edge("chart", END)
    return g.compile()
 
 
GRAPH = build_graph()