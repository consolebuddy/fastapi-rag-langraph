import os
from openai import OpenAI

_client = None

def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("ULTRASAFE_API_KEY"),
            base_url=os.getenv("ULTRASAFE_BASE_URL"),
        )
    return _client

CHAT_MODEL = os.getenv("ULTRASAFE_CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("ULTRASAFE_EMBED_MODEL", "text-embedding-3-large")

async def embed(texts: list[str]) -> list[list[float]]:
    resp = client().embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

async def chat_completion(messages: list[dict], max_tokens: int = 512) -> str:
    resp = client().chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content