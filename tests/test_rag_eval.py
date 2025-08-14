import pytest
from rag.eval import LabeledQuery, eval_retrieval

@pytest.mark.asyncio
async def test_eval_smoke():
    qs = [LabeledQuery(q="reset password", expected_ids=["faq.md:0"]) ]
    res = await eval_retrieval(qs, k=5)
    assert "recall@k" in res