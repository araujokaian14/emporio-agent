"""Policy chunking and retrieval tests. Uses a fake embedder, so no Ollama is needed."""

import numpy as np
import pytest

from emporio_agent.policies_rag import PolicyIndex, extract_text, split_sections
from emporio_agent.config import DATA_DIR


@pytest.fixture(scope="module")
def chunks():
    return split_sections(extract_text(next(DATA_DIR.glob("*.pdf"))))


def test_manual_is_split_by_its_own_sections(chunks):
    sections = [c.section for c in chunks]
    for expected in ["2", "3.1", "4.1", "4.2", "5.2", "6.2", "7.3", "8.1", "9"]:
        assert expected in sections
    assert len(sections) == len(set(sections))  # no duplicated sections


def test_numbered_list_is_not_mistaken_for_heading(chunks):
    flow = next(c for c in chunks if c.section == "7.2")
    assert "Saudação" in flow.text and "Fechamento" in flow.text


def test_headers_and_footers_are_removed(chunks):
    lines = [l for c in chunks for l in c.text.splitlines()]
    assert not any(l.startswith("Página") for l in lines)
    assert "Empório da Música Manual de Políticas e Procedimentos" not in lines


@pytest.mark.parametrize("question,section", [
    ("Me arrependi da compra, posso devolver?", "4.1"),
    ("Qual o horário de sábado?", "2"),
    ("Dá pra parcelar em 10x no cartão?", "3.1"),
    ("Meu violão veio com defeito", "4.2"),
])
def test_keyword_retrieval_finds_right_section(chunks, question, section):
    index = PolicyIndex(chunks, embedder=None)
    top3 = [c.section for c, _ in index.search(question, k=3)]
    assert section in top3


def _fake_embedder(texts):
    """Deterministic bag-of-words hashing vectors: enough to test the hybrid plumbing."""
    vecs = np.zeros((len(texts), 256), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().split():
            vecs[i, hash(w[:5]) % 256] += 1
    return vecs


def test_hybrid_retrieval_with_embeddings(chunks):
    index = PolicyIndex(chunks, embedder=_fake_embedder)
    result = index.search_policies("prazo de entrega SEDEX para outra cidade")
    assert result["retrieval"] == "hybrid"
    assert result["excerpts"][0]["section"] in {"5.2", "5.1"}


def test_falls_back_to_keywords_when_embedding_model_is_missing(chunks):
    def broken(_):
        raise ConnectionError("ollama not running")

    index = PolicyIndex(chunks, embedder=broken)
    result = index.search_policies("posso devolver?")
    assert result["retrieval"] == "keyword" and result["excerpts"]
