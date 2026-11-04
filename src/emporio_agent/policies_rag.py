"""Retrieval over the store policy manual (the only free-text source, so the only RAG part).

Pipeline:
1. Extract the PDF text with pypdf (layout mode keeps tables and bullets readable).
2. Remove page headers/footers and split by the manual's own numbering (2., 4.1, 4.2...).
   Each subsection becomes one chunk, prefixed with its parent title for context.
   The manual is small (~25 chunks), so section-based chunking beats fixed-size windows:
   a chunk never cuts a rule in half.
3. Embed chunks with a local embedding model (Ollama `nomic-embed-text`) and cache them on disk.
4. Hybrid scoring: 0.8 * cosine similarity + 0.2 * keyword overlap. The lexical part helps
   with exact terms ("PIX", "SEDEX", "boleto") that embeddings sometimes under-weight.
   If the embedding model is unavailable, retrieval degrades to keyword-only instead of failing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .config import DATA_DIR, EMBED_MODEL, LLM_API_KEY, LLM_BASE_URL, PROJECT_ROOT
from .data_loader import normalize

log = logging.getLogger(__name__)

Embedder = Callable[[Sequence[str]], np.ndarray]

HEADER_RE = re.compile(r"^\s*Empório da Música Manual de Políticas e Procedimentos\s*$")
FOOTER_RE = re.compile(r"^\s*Página\s+\d+\s*$")
TOP_RE = re.compile(r"^(\d{1,2})\.\s+(\S.*)$")          # "4. Política de Trocas" (column 0)
SUB_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})\s+(\S.*)$")  # "4.1 Direito de Arrependimento"
STOPWORDS = {"a", "o", "e", "de", "da", "do", "das", "dos", "em", "no", "na", "para", "por",
             "com", "um", "uma", "os", "as", "que", "se", "eu", "meu", "minha", "voces", "vcs",
             "qual", "quais", "como", "posso", "pode", "tem", "ter", "sobre"}


@dataclass
class Chunk:
    section: str   # "4.1"
    title: str     # "Direito de Arrependimento (Compras Online) — Política de Trocas e Devoluções"
    text: str

    def as_context(self) -> str:
        return f"[Seção {self.section}] {self.title}\n{self.text}"


# --------------------------------------------------------------------------- PDF -> chunks

def extract_text(pdf_path: Path) -> str:
    from pypdf import PdfReader

    pages = [p.extract_text(extraction_mode="layout") for p in PdfReader(str(pdf_path)).pages]
    lines = []
    for line in "\n".join(pages).replace("​", "").splitlines():
        if HEADER_RE.match(line) or FOOTER_RE.match(line):
            continue
        line = line.replace("•", "-")
        line = re.sub(r"\s{3,}", " | ", line.strip())  # table columns -> "a | b | c"
        lines.append(line)
    return "\n".join(lines)


def split_sections(text: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    top_num, top_title = 0, "Capa e contatos"
    section, title, body = "0", top_title, []

    def flush():
        content = "\n".join(l for l in body if l).strip()
        if content:
            chunks.append(Chunk(section, title, content))

    for line in text.splitlines():
        sub, top = SUB_RE.match(line), TOP_RE.match(line)
        # Only accept headings that follow the manual's numbering, so numbered
        # lists inside a section (e.g. "1. Saudação") are not mistaken for headings.
        if sub and int(sub.group(1)) == top_num:
            flush()
            section = f"{sub.group(1)}.{sub.group(2)}"
            title, body = f"{sub.group(3).strip()} — {top_title}", []
        elif top and int(top.group(1)) == top_num + 1:
            flush()
            top_num, top_title = int(top.group(1)), top.group(2).strip()
            section, title, body = str(top_num), top_title, []
        else:
            body.append(line)
    flush()
    return chunks


# --------------------------------------------------------------------------- embeddings

def ollama_embedder(model: str = EMBED_MODEL) -> Embedder:
    from openai import OpenAI

    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    def embed(texts: Sequence[str]) -> np.ndarray:
        resp = client.embeddings.create(model=model, input=list(texts))
        return np.array([d.embedding for d in resp.data], dtype=np.float32)

    return embed


def _unit(m: np.ndarray) -> np.ndarray:
    return m / np.clip(np.linalg.norm(m, axis=-1, keepdims=True), 1e-9, None)


def _tokens(text: str) -> set[str]:
    """Crude Portuguese stemming: keep the first 5 letters, so 'arrependi' matches
    'arrependimento' and 'devolver' matches 'devolução'. Good enough for ~25 chunks."""
    return {t[:5] for t in normalize(text).split() if len(t) > 1 and t not in STOPWORDS}


# --------------------------------------------------------------------------- index

class PolicyIndex:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None,
                 cache_dir: Path | None = None, model_name: str = EMBED_MODEL):
        self.chunks = chunks
        self.embedder = embedder
        self.model_name = model_name
        self._doc_prefix, self._query_prefix = (
            ("search_document: ", "search_query: ") if "nomic" in model_name else ("", ""))
        self._chunk_tokens = [_tokens(c.as_context()) for c in chunks]
        self.vectors = self._build_vectors(cache_dir) if embedder else None

    @classmethod
    def from_pdf(cls, pdf_path: Path | None = None, embedder: Embedder | None = None,
                 use_embeddings: bool = True) -> "PolicyIndex":
        pdf_path = pdf_path or next(Path(DATA_DIR).glob("*.pdf"))
        chunks = split_sections(extract_text(pdf_path))
        if embedder is None and use_embeddings:
            embedder = ollama_embedder()
        return cls(chunks, embedder, cache_dir=PROJECT_ROOT / ".index")

    def _build_vectors(self, cache_dir: Path | None) -> np.ndarray | None:
        texts = [self._doc_prefix + c.as_context() for c in self.chunks]
        key = hashlib.sha256((self.model_name + json.dumps(texts)).encode()).hexdigest()[:16]
        cache = cache_dir / f"policies_{key}.npy" if cache_dir else None
        if cache and cache.exists():
            return np.load(cache)
        try:
            vectors = _unit(self.embedder(texts))
        except Exception as exc:  # model not pulled / server down -> keyword-only retrieval
            log.warning("Embeddings unavailable (%s). Falling back to keyword retrieval.", exc)
            self.embedder = None
            return None
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, vectors)
        return vectors

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        q_tokens = _tokens(query)
        lexical = np.array([len(q_tokens & ct) / (len(q_tokens) or 1) for ct in self._chunk_tokens])
        scores = lexical
        if self.vectors is not None and self.embedder is not None:
            try:
                q = _unit(self.embedder([self._query_prefix + query]))[0]
                scores = 0.8 * (self.vectors @ q) + 0.2 * lexical
            except Exception as exc:
                log.warning("Query embedding failed (%s); using keyword scores.", exc)
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def search_policies(self, question: str, k: int = 3) -> dict:
        """Tool entry point: returns the most relevant policy excerpts for the question."""
        k = max(1, min(int(k), 5))
        return {
            "retrieval": "hybrid" if self.vectors is not None else "keyword",
            "excerpts": [{"section": c.section, "title": c.title, "text": c.text, "score": round(s, 3)}
                         for c, s in self.search(question, k)],
        }
