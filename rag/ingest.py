import argparse, os, glob
from pathlib import Path
from .llm import embed
import chromadb
from chromadb.utils import embedding_functions

async def _read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        # light PDF text extraction without heavy deps
        try:
            import pypdf
            pdf = pypdf.PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception:
            return ""
    return Path(path).read_text(encoding="utf-8", errors="ignore")

def chunk(text: str, size=1200, overlap=150):
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i+size])
        i += max(1, size-overlap)
    return out

async def build_index(docs_dir: str, persist_dir: str):
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    col = client.get_or_create_collection("support_docs")

    files = [Path(p) for p in glob.glob(os.path.join(docs_dir, "**/*"), recursive=True) if os.path.isfile(p)]
    ids, texts, metas = [], [], []

    for f in files:
        txt = await _read_text(f)
        for j, ch in enumerate(chunk(txt)):
            ids.append(f"{f.name}:{j}")
            texts.append(ch)
            metas.append({"source": f.name, "path": str(f)})

    if texts:
        # embed in batches
        B=64
        vectors = []
        for i in range(0, len(texts), B):
            vectors.extend(await embed(texts[i:i+B]))
        col.add(ids=ids, documents=texts, metadatas=metas, embeddings=vectors)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--persist", default="./storage/chroma")
    args = ap.parse_args()
    import asyncio
    asyncio.run(build_index(args.docs, args.persist))