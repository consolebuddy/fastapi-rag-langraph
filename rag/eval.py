from dataclasses import dataclass
from typing import List, Dict
import numpy as np
from .retriever import dense_search, bm25_search, rrf_merge, rerank

@dataclass
class LabeledQuery:
    q: str
    expected_ids: List[str]

async def eval_retrieval(queries: List[LabeledQuery], k: int = 10) -> Dict:
    recalls, mrrs = [], []
    for item in queries:
        dense = await dense_search(item.q, k)
        sparse = await bm25_search(item.q, k)
        fused = rrf_merge(dense, sparse, k=50)
        reranked = await rerank(item.q, fused, top_n=k)
        got_ids = [c["id"] for c in reranked]
        # Recall@k
        hit = len(set(got_ids) & set(item.expected_ids)) / max(1, len(item.expected_ids))
        recalls.append(hit)
        # MRR
        rr = 0.0
        for i, gid in enumerate(got_ids, start=1):
            if gid in item.expected_ids:
                rr = 1.0/i
                break
        mrrs.append(rr)
    return {"recall@k": float(np.mean(recalls)), "mrr@k": float(np.mean(mrrs)), "n": len(queries)}