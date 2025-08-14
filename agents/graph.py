from typing import Optional
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from .nodes import research_node, summarize_node, critic_node, writer_node

class ResearchState(BaseModel):
    question: str
    retrieved: list | None = None
    summary: str | None = None
    critique: str | None = None
    report: str | None = None

async def run_research(question: str, max_papers: Optional[int] = None):
    g = StateGraph(ResearchState)
    g.add_node("research", research_node)
    g.add_node("summarize", summarize_node)
    g.add_node("critic", critic_node)
    g.add_node("writer", writer_node)

    g.set_entry_point("research")
    g.add_edge("research", "summarize")
    g.add_edge("summarize", "critic")
    g.add_edge("critic", "writer")
    g.add_edge("writer", END)

    app = g.compile()
    initial = ResearchState(question=question)
    final = await app.ainvoke(initial)
    return {"report": final.report, "chunks": len(final.retrieved or [])}