"""Wrapper de embeddings usando Sentence Transformers.

Gera embeddings multilíngues (português) para os chunks do CDC e para as
perguntas do usuário, compatível com LangChain e pgvector.
"""
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL, EMBEDDING_DIM


class STEmbeddings(Embeddings):
    """Implementação de Embeddings do LangChain sobre Sentence Transformer."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        )
        return vector[0].tolist()


@lru_cache(maxsize=1)
def get_embeddings() -> STEmbeddings:
    return STEmbeddings(EMBEDDING_MODEL)


if __name__ == "__main__":
    eb = get_embeddings()
    v = eb.embed_query("Qual o prazo para reclamação do vício do produto?")
    print(f"Embedding dim: {len(v)} (esperado {EMBEDDING_DIM})")
