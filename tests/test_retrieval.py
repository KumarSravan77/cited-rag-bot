from app.models import Chunk
from app.retrieval import HashEmbedder, hybrid_search


def chunks():
    embedder = HashEmbedder()
    texts = [
        "The refund policy allows returns within thirty days.",
        "Database backups run every six hours.",
        "Support is available Monday through Friday.",
    ]
    return [
        Chunk(document="handbook.pdf", page=index, text=text, embedding=vector)
        for index, (text, vector) in enumerate(zip(texts, embedder.embed(texts)), start=1)
    ]


def test_hybrid_search_returns_relevant_page_first():
    hits = hybrid_search("When can I return a purchase?", chunks(), HashEmbedder(), 2)
    assert hits[0].page == 1
    assert "refund" in hits[0].text.lower()


def test_results_keep_page_level_provenance():
    hit = hybrid_search("How often are backups?", chunks(), HashEmbedder(), 1)[0]
    assert hit.document == "handbook.pdf"
    assert hit.page == 2
