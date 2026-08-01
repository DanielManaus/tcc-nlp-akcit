"""Pipeline de ingestão do CDC.

1. Obtém o texto do CDC (arquivo local ou download do Planalto via Docling).
2. Segmenta em chunks por artigo (estratégia do TCC).
3. Gera embeddings (Sentence Transformers).
4. Indexa no PostgreSQL + pgvector (idempotente: não duplica).

Execução: python -m src.ingestion
"""
import os
import re
import time

from langchain_core.documents import Document
from langchain_postgres import PGVector

from src.config import (
    CDC_LOCAL_PATH,
    CDC_URL,
    COLLECTION_NAME,
    HISTORY_COLLECTION_NAME,
    HISTORY_LOCAL_PATH,
    JURISPRUDENCE_COLLECTION_NAME,
    STJ_LOCAL_PATH,
    connection_string,
    psycopg_connection_string,
)
from src.embeddings import get_embeddings


def _download_html_to_markdown() -> str:
    """Extrai texto de páginas HTML oficiais com retry e User-Agent."""
    import requests
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": "Mozilla/5.0 (RAG-CDC-POC; academic project)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = requests.get(
                CDC_URL,
                headers=headers,
                timeout=(10, 90),
            )
            response.raise_for_status()
            encoding = (
                "iso-8859-1"
                if "planalto.gov.br" in CDC_URL.lower()
                else response.apparent_encoding or "utf-8"
            )
            html = response.content.decode(encoding, errors="replace")
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            text = soup.get_text("\n")
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            with open(CDC_LOCAL_PATH, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            print(f"✓ CDC salvo a partir do HTML oficial: {CDC_LOCAL_PATH}")
            return CDC_LOCAL_PATH
        except Exception as exc:  # pragma: no cover
            last_error = exc
            print(f"⚠️ Tentativa HTML {attempt}/3 falhou: {exc}")
            time.sleep(2 * attempt)

    raise RuntimeError(f"Falha ao baixar HTML oficial do CDC: {last_error}")


def _ensure_corpus() -> str:
    """Baixa/converte o CDC oficial se não houver corpus local."""
    if os.path.exists(CDC_LOCAL_PATH):
        print(f"✓ Corpus local encontrado: {CDC_LOCAL_PATH}")
        return CDC_LOCAL_PATH

    os.makedirs(os.path.dirname(CDC_LOCAL_PATH), exist_ok=True)
    print(f"⬇️  Baixando CDC oficial de {CDC_URL} ...")

    if CDC_URL.lower().endswith((".htm", ".html")):
        return _download_html_to_markdown()

    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(CDC_URL)
        md = result.document.export_to_markdown()
        with open(CDC_LOCAL_PATH, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"✓ CDC convertido para markdown: {CDC_LOCAL_PATH}")
        return CDC_LOCAL_PATH
    except Exception as exc:  # pragma: no cover
        print(f"⚠️ Docling falhou ({exc}). Tentando extração HTML simples...")

    try:
        return _download_html_to_markdown()
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"Não foi possível obter o CDC. Coloque o texto em {CDC_LOCAL_PATH}. ({exc})"
        )


def _read_text(path: str) -> str:
    if path.endswith(".md"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(path)
    return result.document.export_to_markdown()


# Padrão para artigos: "Art. 1º", "Art. 6ş", "Art. 54-A", "Artigo 3" etc.
ART_PATTERN = re.compile(
    r"(?im)^\s*(?:Art(?:igo)?\.?\s*(\d+(?:-[A-Z])?)\s*[º°ªş]?(?=\.|\s|$))",
)


def split_by_article(text: str) -> list[Document]:
    """Segmenta o texto do CDC por artigo, mantendo metadados de número."""
    lines = [ln.rstrip() for ln in text.splitlines()]
    chunks: list[Document] = []
    current_article = "Preâmbulo"
    buffer: list[str] = []

    def flush(art: str, buf: list[str]):
        content = "\n".join(buf).strip()
        if content:
            chunks.append(
                Document(page_content=content, metadata={"artigo": art})
            )

    for ln in lines:
        m = ART_PATTERN.match(ln)
        if m:
            flush(current_article, buffer)
            buffer = [ln]
            current_article = f"Art. {m.group(1).upper()}"
        else:
            buffer.append(ln)
    flush(current_article, buffer)

    # Se não segmentou nada (sem padrão), fallback por tamanho
    if len(chunks) <= 1:
        chunks = []
        for i, piece in enumerate(re.split(r"\n{2,}", text)):
            piece = piece.strip()
            if piece:
                chunks.append(
                    Document(
                        page_content=piece,
                        metadata={"artigo": f"Trecho {i + 1}"},
                    )
                )
    print(f"✓ Segmentado em {len(chunks)} chunks (por artigo).")
    return chunks


STJ_ENTRY_PATTERN = re.compile(r"(?m)^##\s+(Súmula\s+\d+)\s*$")


HISTORY_ENTRY_PATTERN = re.compile(r"(?m)^##\s+(.+?)\s*$")


def split_stj_summaries(text: str) -> list[Document]:
    """Segmenta o corpus curado de súmulas do STJ."""
    matches = list(STJ_ENTRY_PATTERN.finditer(text))
    documents: list[Document] = []

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        referencia = match.group(1)
        tema_match = re.search(r"(?m)^Tema:\s*(.+)$", content)
        fonte_match = re.search(r"(?m)^Fonte oficial:\s*(.+)$", content)
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "fonte": "STJ",
                    "tipo": "jurisprudencia",
                    "referencia": referencia,
                    "tema": tema_match.group(1) if tema_match else "",
                    "url": fonte_match.group(1) if fonte_match else "",
                },
            )
        )

    print(f"✓ Segmentado em {len(documents)} entradas de jurisprudência STJ.")
    return documents


def split_history_notes(text: str) -> list[Document]:
    """Segmenta notas historicas/institucionais curadas sobre o CDC."""
    matches = list(HISTORY_ENTRY_PATTERN.finditer(text))
    documents: list[Document] = []

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        titulo = match.group(1).strip()
        tema_match = re.search(r"(?m)^Tema:\s*(.+)$", content)
        fonte_match = re.search(r"(?m)^Fonte oficial:\s*(.+)$", content)
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "fonte": "Historico CDC",
                    "tipo": "historico",
                    "referencia": titulo,
                    "tema": tema_match.group(1) if tema_match else titulo,
                    "url": fonte_match.group(1) if fonte_match else "",
                },
            )
        )

    print(f"✓ Segmentado em {len(documents)} entradas historicas do CDC.")
    return documents


def _collection_count(conn_str: str, collection_name: str = COLLECTION_NAME) -> int:
    """Retorna a quantidade de registros já indexados (0 se não existir)."""
    try:
        from psycopg import connect

        conn = connect(conn_str)
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM langchain_pg_embedding "
            "WHERE collection_id = (SELECT uuid FROM langchain_pg_collection "
            "WHERE name = %s);",
            (collection_name,),
        )
        count = cur.fetchone()[0]
        conn.close()
        return int(count)
    except Exception:
        return 0


def _delete_collection(conn_str: str, collection_name: str = COLLECTION_NAME) -> None:
    """Remove a coleção atual e seus embeddings para reindexação limpa."""
    try:
        from psycopg import connect

        with connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM langchain_pg_embedding "
                    "WHERE collection_id = (SELECT uuid FROM langchain_pg_collection "
                    "WHERE name = %s);",
                    (collection_name,),
                )
                cur.execute(
                    "DELETE FROM langchain_pg_collection WHERE name = %s;",
                    (collection_name,),
                )
            conn.commit()
    except Exception:
        # Tabelas podem ainda não existir na primeira execução.
        pass


def _index_documents(
    documents: list[Document],
    collection_name: str,
    label: str,
    force_recreate: bool = False,
) -> None:
    embeddings = get_embeddings()

    if force_recreate:
        _delete_collection(psycopg_connection_string(), collection_name)

    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection_string(),
        use_jsonb=True,
    )

    if not force_recreate:
        existing = _collection_count(psycopg_connection_string(), collection_name)
        if existing > 0:
            print(
                f"✓ {label} já indexado ({existing} chunks). Pulando ingestão. "
                "Use force_recreate=True para reindexar."
            )
            return

    vectorstore.add_documents(documents)
    print(f"✓ Indexado {len(documents)} chunks em '{collection_name}' ({label}).")


def ingest_stj_jurisprudence(force_recreate: bool = False) -> None:
    if not os.path.exists(STJ_LOCAL_PATH):
        print(f"⚠️ Corpus STJ não encontrado em {STJ_LOCAL_PATH}. Pulando.")
        return
    text = _read_text(STJ_LOCAL_PATH)
    documents = split_stj_summaries(text)
    _index_documents(
        documents,
        JURISPRUDENCE_COLLECTION_NAME,
        "jurisprudência STJ",
        force_recreate=force_recreate,
    )


def ingest_history_context(force_recreate: bool = False) -> None:
    if not os.path.exists(HISTORY_LOCAL_PATH):
        print(f"⚠️ Corpus historico não encontrado em {HISTORY_LOCAL_PATH}. Pulando.")
        return
    text = _read_text(HISTORY_LOCAL_PATH)
    documents = split_history_notes(text)
    _index_documents(
        documents,
        HISTORY_COLLECTION_NAME,
        "historico CDC",
        force_recreate=force_recreate,
    )


def ingest(force_recreate: bool = False, refresh_corpus: bool = False) -> None:
    if refresh_corpus and os.path.exists(CDC_LOCAL_PATH):
        os.remove(CDC_LOCAL_PATH)
    path = _ensure_corpus()
    text = _read_text(path)
    documents = split_by_article(text)

    _index_documents(
        documents,
        COLLECTION_NAME,
        "CDC",
        force_recreate=force_recreate,
    )
    ingest_stj_jurisprudence(force_recreate=force_recreate)
    ingest_history_context(force_recreate=force_recreate)


if __name__ == "__main__":
    ingest()
