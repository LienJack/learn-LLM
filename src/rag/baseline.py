from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    source: str = "local"

    def __post_init__(self) -> None:
        if not self.doc_id:
            raise ValueError("doc_id must not be empty.")
        if not self.text:
            raise ValueError("text must not be empty.")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    start: int
    end: int
    source: str


@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Citation:
    doc_id: str
    chunk_id: str
    title: str
    source: str
    start: int
    end: int


@dataclass(frozen=True)
class RAGAnswer:
    answer: str
    citations: list[Citation]
    retrieved: list[RetrievalResult]
    refused: bool


def chunk_document(document: Document, chunk_size: int, overlap: int = 0) -> list[Chunk]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size).")

    chunks: list[Chunk] = []
    start = 0
    chunk_index = 0
    step = chunk_size - overlap
    while start < len(document.text):
        end = min(start + chunk_size, len(document.text))
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}#chunk_{chunk_index:03d}",
                doc_id=document.doc_id,
                title=document.title,
                text=document.text[start:end],
                start=start,
                end=end,
                source=document.source,
            ),
        )
        if end == len(document.text):
            break
        start += step
        chunk_index += 1
    return chunks


def chunk_documents(documents: list[Document], chunk_size: int, overlap: int = 0) -> list[Chunk]:
    return [
        chunk
        for document in documents
        for chunk in chunk_document(document, chunk_size=chunk_size, overlap=overlap)
    ]


class HashingTextEmbedder:
    """A deterministic character hashing embedder for local RAG tests."""

    def __init__(self, dim: int = 128) -> None:
        if dim < 1:
            raise ValueError("dim must be positive.")
        self.dim = dim

    def embed(self, text: str) -> Tensor:
        vector = torch.zeros(self.dim, dtype=torch.float32)
        for char in text.lower():
            if char.isspace():
                continue
            vector[ord(char) % self.dim] += 1.0
        norm = vector.norm()
        return vector / norm if norm > 0 else vector


class VectorStore:
    def __init__(self, chunks: list[Chunk], embedder: HashingTextEmbedder) -> None:
        if not chunks:
            raise ValueError("chunks must not be empty.")
        self.chunks = chunks
        self.embedder = embedder
        self.embeddings = torch.stack([embedder.embed(chunk.text) for chunk in chunks])

    def search(self, query: str, top_k: int, min_score: float = 0.0) -> list[RetrievalResult]:
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        query_embedding = self.embedder.embed(query)
        scores = self.embeddings @ query_embedding
        k = min(top_k, len(self.chunks))
        values, indices = torch.topk(scores, k=k)
        results: list[RetrievalResult] = []
        for score, index in zip(values.tolist(), indices.tolist(), strict=True):
            if score >= min_score:
                results.append(RetrievalResult(chunk=self.chunks[index], score=float(score)))
        return results


def build_context_prompt(query: str, results: list[RetrievalResult]) -> str:
    lines = [
        "你只能基于给定资料回答。",
        "如果资料不足，请说“资料不足，无法判断”。",
        "回答中必须引用来源编号。",
        "",
    ]
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        lines.extend(
            [
                (
                    f"[资料 {index}] doc_id={chunk.doc_id} chunk_id={chunk.chunk_id} "
                    f"title={chunk.title} score={result.score:.4f}"
                ),
                chunk.text,
                "",
            ],
        )
    lines.append(f"问题：{query}")
    return "\n".join(lines)


def citation_from_chunk(chunk: Chunk) -> Citation:
    return Citation(
        doc_id=chunk.doc_id,
        chunk_id=chunk.chunk_id,
        title=chunk.title,
        source=chunk.source,
        start=chunk.start,
        end=chunk.end,
    )


def answer_with_citations(
    query: str,
    results: list[RetrievalResult],
    min_score: float = 0.1,
) -> RAGAnswer:
    if not results or results[0].score < min_score:
        return RAGAnswer(
            answer="资料不足，无法判断。",
            citations=[],
            retrieved=results,
            refused=True,
        )

    best = results[0].chunk
    citation = citation_from_chunk(best)
    answer = f"根据 {citation.chunk_id}，{best.text}"
    return RAGAnswer(
        answer=answer,
        citations=[citation],
        retrieved=results,
        refused=False,
    )


def run_rag(
    query: str,
    store: VectorStore,
    top_k: int = 3,
    min_score: float = 0.1,
) -> RAGAnswer:
    results = store.search(query, top_k=top_k, min_score=0.0)
    return answer_with_citations(query, results, min_score=min_score)


def validate_citations(answer: RAGAnswer, chunks: list[Chunk]) -> None:
    chunk_ids = {chunk.chunk_id for chunk in chunks}
    missing = [
        citation.chunk_id
        for citation in answer.citations
        if citation.chunk_id not in chunk_ids
    ]
    if missing:
        raise ValueError(f"citations point to missing chunks: {missing}")
