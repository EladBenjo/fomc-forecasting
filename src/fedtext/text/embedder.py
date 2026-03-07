"""
Embedder stub — to be implemented when a vector store is chosen.

Intended interface:
  run(conn, model) -> embeds all chunks in the DB that have embedding=NULL
                      and writes the vector blob back to chunks.embedding

Model options to evaluate:
  - sentence-transformers (local, e.g. all-MiniLM-L6-v2)
  - OpenAI text-embedding-3-small (API, best quality)
  - ROBER-FOMC fine-tuned model (domain-specific, CC-BY-NC-4.0)

Vector store options:
  - sqlite-vec  (stays in SQLite, zero new services, good to ~100k chunks)
  - ChromaDB    (easy LangChain/LlamaIndex integration)
  - FAISS       (fastest similarity search, no persistence)
"""


def run(*args, **kwargs) -> None:
    raise NotImplementedError(
        "Embedder not yet implemented. "
        "Choose an embedding model and vector store, then implement this module."
    )
