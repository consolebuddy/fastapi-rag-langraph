from rag.retriever import dense_search, bm25_search, rrf_merge, rerank
from rag.llm import chat_completion
from .compression import budget_chunks

async def research_node(state):
    q = state["question"]
    dense = await dense_search(q, 12)
    sparse = await bm25_search(q, 12)
    fused = rrf_merge(dense, sparse, 50)
    top = await rerank(q, fused, 12)
    state["retrieved"] = top
    return state

async def summarize_node(state):
    chunks = budget_chunks(state["retrieved"], 8000)
    ctx = "\n\n".join([c["text"] for c in chunks])
    prompt = [
        {"role":"system","content":"Extract key claims, methods, datasets, and limitations. Use bullet points."},
        {"role":"user","content": f"Context: {ctx}"}
    ]
    summary = await chat_completion(prompt, max_tokens=700)
    state["summary"] = summary
    return state

async def critic_node(state):
    prompt = [
        {"role":"system","content":"You are a critical reviewer. Evaluate the summary for evidence quality, confounds, and recency. Return a JSON with fields: strengths, weaknesses, risks."},
        {"role":"user","content": state["summary"]}
    ]
    critique = await chat_completion(prompt, max_tokens=400)
    state["critique"] = critique
    return state

async def writer_node(state):
    prompt = [
        {"role":"system","content":"Write a concise research brief with sections: Background, Findings, Evidence, Caveats, Recommendations. Add inline bracketed citations if present in context."},
        {"role":"user","content": f"Summary: {state['summary']}\n\nCritique: {state['critique']}"}
    ]
    report = await chat_completion(prompt, max_tokens=900)
    state["report"] = report
    return state