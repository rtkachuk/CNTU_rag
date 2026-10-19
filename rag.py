import os
import argparse
from pathlib import Path

import chromadb
from litellm import embedding, completion

LITELLM_HOST    = os.environ.get("LITELLM_HOST",    "Address for your litellm")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "your-key-here")

EMBED_MODEL = os.environ.get("EMBED_MODEL", "litellm_proxy/qwen-embedding-8b")
CHAT_MODEL  = os.environ.get("CHAT_MODEL",  "litellm_proxy/qwen-instruct-8b")

DB_PATH       = "./chroma_db"
COLLECTION    = "documents"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100
TOP_K         = 5


def embed(texts):
    """Turn a list of texts into embedding vectors using LiteLLM."""
    response = embedding(
        model=EMBED_MODEL,
        input=texts,
        api_base=LITELLM_HOST,
        api_key=LITELLM_API_KEY,
    )
    return [item["embedding"] for item in response.data]


def get_collection():
    db = chromadb.PersistentClient(path=DB_PATH)
    return db.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size].strip())
        start += size - overlap
    return [c for c in chunks if c]


def ingest(path):
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        return

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        chunks = chunk_text(f.read())

    ids   = [f"{Path(path).stem}__{i}" for i in range(len(chunks))]
    metas = [{"source": path, "chunk": i} for i in range(len(chunks))]

    col = get_collection()
    col.upsert(
        ids=ids,
        documents=chunks,
        metadatas=metas,
        embeddings=embed(chunks),
    )
    print(f"Ingested {path} → {len(chunks)} chunks → {DB_PATH}")


def ask(question):
    col = get_collection()
    if col.count() == 0:
        print("Vector store is empty. Ingest some documents first (-i).")
        return

    results = col.query(
        query_embeddings=embed([question]),
        n_results=min(TOP_K, col.count()),
    )
    hits = list(zip(results["documents"][0], results["metadatas"][0]))

    context = "\n\n---\n\n".join(
        f"[Source: {m['source']}, chunk {m['chunk']}]\n{doc}" for doc, m in hits
    )
    prompt = (
        "You are a helpful assistant. Answer ONLY from the provided context.\n"
        "If the answer isn't there, say so — do not invent information.\n"
        "If user asks you to write the code - write it using provided context as a base.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "ANSWER:"
    )

    resp = completion(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        api_base=LITELLM_HOST,
        api_key=LITELLM_API_KEY,
    )
    answer = resp.choices[0].message.content.strip()

    print(f"\n🔍 {question}\n")
    print(answer)
    print("\n📎 Sources:")
    for source in sorted({m["source"] for _, m in hits}):
        print(f"  • {source}")

def main():
    parser = argparse.ArgumentParser(description="Simple RAG app (ChromaDB + LiteLLM).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--ingest", metavar="FILE",
                       help="ingest a single .txt/.md file")
    group.add_argument("-a", "--ask", metavar="QUESTION",
                       help="ask a question")
    args = parser.parse_args()

    if args.ingest:
        ingest(args.ingest)
    else:
        ask(args.ask)


if __name__ == "__main__":
    main()
