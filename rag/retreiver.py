import os, math
import numpy as np
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from .llm import chat_completion, embed

COLLECTION = "support_docs"
CHROMA_DIR = os.getenv("CHROMA_DIR", "./storage/chroma")

_cross = None

def _cross_encoder():
    global _cross
    if _cross is None:
        _cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
    return _cross

async def dense_search(query: str, k: int = 12):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_or_create_collection(COLLECTION)
    qv = (await embed([query]))[0]
    res = col.query(query_embeddings=[qv], n_results=k, include=["documents","metadatas","distances","embeddings"])
    docs = [
        {
            "id": _id,
            "text": doc,
            "meta": meta,
            "score": -dist if dist is not None else 0.0,
        }
        for _id, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]
    return docs

async def bm25_search(query: str, k: int = 12):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_or_create_collection(COLLECTION)
    all_docs = col.get(include=["documents","metadatas","ids"])  # small corpora ok
    corpus = [d for d in all_docs["documents"]]
    bm25 = BM25Okapi([c.split() for c in corpus])
    scores = bm25.get_scores(query.split())
    idx = np.argsort(scores)[::-1][:k]
    return [
        {"id": all_docs["ids"][i], "text": all_docs["documents"][i], "meta": all_docs["metadatas"][i], "score": float(scores[i])}
        for i in idx
    ]

def rrf_merge(dense, sparse, k=25):
    # Reciprocal Rank Fusion
    ranks = {}
    for rank, d in enumerate(dense):
        ranks[d["id"]] = ranks.get(d["id"], 0.0) + 1.0/(60+rank)
    for rank, d in enumerate(sparse):
        ranks[d["id"]] = ranks.get(d["id"], 0.0) + 1.0/(60+rank)
    by_rrf = sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:k]
    # Rehydrate payloads from dense priority, falling back to sparse
    m = {d["id"]: d for d in dense}
    m.update({d["id"]: d for d in sparse})
    return [m[i] for (i, _) in by_rrf]

async def rerank(query: str, cands: list[dict], top_n: int = 6):
    m = _cross_encoder()
    pairs = [(query, c["text"]) for c in cands]
    scores = m.predict(pairs)
    for c, s in zip(cands, scores):
        c["re_score"] = float(s)
    return sorted(cands, key=lambda x: x["re_score"], reverse=True)[:top_n]

def compress(chunks: list[dict], max_chars: int = 10000, mmr_lambda: float = 0.6):
    # Simple MMR-like selection by cosine distance on embeddings if available; else greedy by rerank score
    sel, total = [], 0
    for c in chunks:
        t = c["text"]
        if total + len(t) > max_chars:
            continue
        sel.append(c)
        total += len(t)
    return sel

async def answer_with_rag(query: str) -> tuple[str, list[str]]:
    dense = await dense_search(query)
    sparse = await bm25_search(query)
    fused = rrf_merge(dense, sparse, k=50)
    top = await rerank(query, fused, top_n=6)
    context = "\n\n".join([f"[source:{t['meta']['source']}]\n{t['text']}" for t in top])

    system = {
        "role":"system",
        "content": (
            "You are a support assistant. Answer using ONLY the provided context. "
            "Cite sources like [source:filename] when relevant."
        )
    }
    user = {"role":"user", "content": f"Question: {query}\n\nContext:\n{context}"}
    answer = await chat_completion([system, user])
    cits = list({t['meta']['source'] for t in top})
    return answer, cits